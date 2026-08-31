// submission/_spimi_cpp.cpp — SPIMI index builder in C++ (optional).
//
// Single-pass in-memory indexing with block flushing: accumulate postings in
// memory and, when the posting count crosses a threshold, flush a sorted
// block to a temp file. finalize() k-way merges the blocks, applies the
// truecase df-ratio gate, and writes the compact v2 format consumed by
// NativeIndex and the Python fallback: terms.txt, postings.bin, doclen.bin,
// meta.json, and the optional docids.bin, tf1.bin, early.bin, and
// case_terms.bin sidecars.
//
// Purpose is OOM-safety on large corpora: peak memory is one block plus the
// streaming merge, never the whole postings set. build_index_from_jsonl()
// performs file reading, JSON decoding, worker threading, tokenisation, and
// indexing behind one pybind call. add_document_from_text() and add_document()
// remain exposed for compatibility, tuning, and tests.
//
// Optional + drop-in: retrieve.build_index() uses this if built and falls
// back to InvertedIndex().build()+save_v2() otherwise.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tokenizer.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <dirent.h>
#include <exception>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace py = pybind11;

// Internal-only marker used to pass prefix terms from the fused tokenizer into
// add_document(). Marker tokens are never persisted in the vocabulary.
static const std::string EARLY_PREFIX = "\x01" "early:";

static bool is_early_term(const std::string &term) {
    return term.compare(0, EARLY_PREFIX.size(), EARLY_PREFIX) == 0;
}

static bool has_upper(const std::string &text) {
    return std::any_of(text.begin(), text.end(), [](char c) {
        return c >= 'A' && c <= 'Z';
    });
}

static bool is_blank_line(const std::string &line) {
    return std::all_of(line.begin(), line.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
}

static void ensure_directory(const std::string &path) {
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create directory: " + path);
}

static void remove_directory_tree(const std::string &path) {
    DIR *directory = ::opendir(path.c_str());
    if (!directory) return;
    while (dirent *entry = ::readdir(directory)) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string child = path + "/" + name;
        struct stat info {};
        if (::lstat(child.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
            remove_directory_tree(child);
        else
            std::remove(child.c_str());
    }
    ::closedir(directory);
    ::rmdir(path.c_str());
}

// Minimal standards-compliant JSON decoder for the corpus JSONL records. It
// extracts string-valued doc_id/text fields, handles every JSON string escape
// (including UTF-16 surrogate pairs), tolerates arbitrary field order and
// extra fields, and rejects malformed records instead of silently indexing
// corrupted text.
class JsonCursor {
public:
    explicit JsonCursor(const std::string &source) : source_(source) {}

    void skip_ws() {
        while (pos_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[pos_])))
            pos_++;
    }

    bool consume(char expected) {
        skip_ws();
        if (pos_ < source_.size() && source_[pos_] == expected) {
            pos_++;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected))
            fail(std::string("expected '") + expected + "'");
    }

    std::string parse_string() {
        skip_ws();
        if (pos_ >= source_.size() || source_[pos_] != '"')
            fail("expected JSON string");
        pos_++;
        std::string out;
        while (pos_ < source_.size()) {
            unsigned char c = static_cast<unsigned char>(source_[pos_++]);
            if (c == '"') return out;
            if (c < 0x20) fail("unescaped control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= source_.size()) fail("unterminated escape");
            char esc = source_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = parse_hex4();
                    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                        if (pos_ + 2 > source_.size() ||
                            source_[pos_] != '\\' || source_[pos_ + 1] != 'u')
                            fail("high surrogate without low surrogate");
                        pos_ += 2;
                        uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF)
                            fail("invalid low surrogate");
                        codepoint = 0x10000 +
                            ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                    } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                        fail("unexpected low surrogate");
                    }
                    append_utf8(out, codepoint);
                    break;
                }
                default:
                    fail("invalid string escape");
            }
        }
        fail("unterminated JSON string");
    }

    void skip_value() {
        skip_ws();
        if (pos_ >= source_.size()) fail("expected JSON value");
        char c = source_[pos_];
        if (c == '"') {
            parse_string();
        } else if (c == '{') {
            pos_++;
            skip_ws();
            if (consume('}')) return;
            while (true) {
                parse_string();
                expect(':');
                skip_value();
                if (consume('}')) return;
                expect(',');
            }
        } else if (c == '[') {
            pos_++;
            skip_ws();
            if (consume(']')) return;
            while (true) {
                skip_value();
                if (consume(']')) return;
                expect(',');
            }
        } else if (c == 't') {
            consume_literal("true");
        } else if (c == 'f') {
            consume_literal("false");
        } else if (c == 'n') {
            consume_literal("null");
        } else {
            skip_number();
        }
    }

    bool at_end() {
        skip_ws();
        return pos_ == source_.size();
    }

private:
    const std::string &source_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string &message) const {
        throw std::runtime_error(
            "invalid JSON at byte " + std::to_string(pos_) + ": " + message);
    }

    static int hex_value(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > source_.size()) fail("truncated unicode escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; i++) {
            int digit = hex_value(source_[pos_++]);
            if (digit < 0) fail("invalid unicode escape");
            value = (value << 4) | static_cast<uint32_t>(digit);
        }
        return value;
    }

    static void append_utf8(std::string &out, uint32_t codepoint) {
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    void consume_literal(const char *literal) {
        size_t begin = pos_;
        while (*literal != '\0') {
            if (pos_ >= source_.size() || source_[pos_] != *literal)
                fail("invalid literal");
            pos_++;
            literal++;
        }
        if (pos_ == begin) fail("invalid literal");
    }

    void skip_number() {
        size_t begin = pos_;
        if (source_[pos_] == '-') pos_++;
        if (pos_ >= source_.size()) fail("invalid number");
        if (source_[pos_] == '0') {
            pos_++;
        } else if (source_[pos_] >= '1' && source_[pos_] <= '9') {
            while (pos_ < source_.size() &&
                   source_[pos_] >= '0' && source_[pos_] <= '9')
                pos_++;
        } else {
            fail("invalid number");
        }
        if (pos_ < source_.size() && source_[pos_] == '.') {
            pos_++;
            size_t fraction = pos_;
            while (pos_ < source_.size() &&
                   source_[pos_] >= '0' && source_[pos_] <= '9')
                pos_++;
            if (pos_ == fraction) fail("invalid number fraction");
        }
        if (pos_ < source_.size() &&
            (source_[pos_] == 'e' || source_[pos_] == 'E')) {
            pos_++;
            if (pos_ < source_.size() &&
                (source_[pos_] == '+' || source_[pos_] == '-'))
                pos_++;
            size_t exponent = pos_;
            while (pos_ < source_.size() &&
                   source_[pos_] >= '0' && source_[pos_] <= '9')
                pos_++;
            if (pos_ == exponent) fail("invalid number exponent");
        }
        if (pos_ == begin) fail("invalid number");
    }
};

static std::pair<std::string, std::string>
parse_corpus_record(const std::string &line) {
    JsonCursor json(line);
    json.expect('{');
    std::string doc_id;
    std::string text;
    bool have_doc_id = false;
    bool have_text = false;
    if (!json.consume('}')) {
        while (true) {
            std::string key = json.parse_string();
            json.expect(':');
            if (key == "doc_id") {
                doc_id = json.parse_string();
                have_doc_id = true;
            } else if (key == "text") {
                text = json.parse_string();
                have_text = true;
            } else {
                json.skip_value();
            }
            if (json.consume('}')) break;
            json.expect(',');
        }
    }
    if (!json.at_end())
        throw std::runtime_error("trailing content after JSON object");
    if (!have_doc_id || !have_text)
        throw std::runtime_error("record must contain string doc_id and text fields");
    return {std::move(doc_id), std::move(text)};
}

static void put_vbyte(std::string &out, long n) {
    while (true) {
        unsigned char b = n & 0x7F;
        n >>= 7;
        if (n) {
            out.push_back(static_cast<char>(b));
        } else {
            out.push_back(static_cast<char>(b | 0x80));
            break;
        }
    }
}

static bool pack_docid(const std::string &id, uint64_t &value) {
    if (id.size() != 8) return false;
    value = 0;
    for (char c : id) {
        int digit = -1;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'z') digit = c - 'a' + 10;
        else return false;
        value = value * 36u + static_cast<uint64_t>(digit);
    }
    return true;
}

static void compact_docids(const std::string &dir) {
    std::ifstream in(dir + "/docs.txt");
    std::vector<uint64_t> values;
    std::string id;
    bool packable = true;
    while (std::getline(in, id)) {
        uint64_t value = 0;
        if (!pack_docid(id, value)) {
            packable = false;
            break;
        }
        values.push_back(value);
    }
    if (!packable) {
        std::remove((dir + "/docids.bin").c_str());
        return;
    }
    std::ofstream out(dir + "/docids.bin", std::ios::binary);
    out.write("D36", 3);
    out.put(static_cast<char>(1));
    for (uint64_t value : values) {
        for (int i = 0; i < 6; i++) {
            out.put(static_cast<char>(value & 0xFFu));
            value >>= 8;
        }
    }
    out.close();
    std::remove((dir + "/docs.txt").c_str());
}

// Append one front-coded term record to `out` (terms.txt is front-coded and
// alphabetical): VByte(shared prefix len with prev) VByte(suffix len) suffix
// VByte(df); updates prev. Mirrors indexer._encode_terms and the terms reader
// in _index_cpp.cpp.
static void put_term_frontcoded(std::string &out, std::string &prev,
                                const std::string &term, long df) {
    size_t s = 0, mm = std::min(prev.size(), term.size());
    while (s < mm && prev[s] == term[s]) s++;
    put_vbyte(out, static_cast<long>(s));
    put_vbyte(out, static_cast<long>(term.size() - s));
    out.append(term, s, term.size() - s);
    put_vbyte(out, df);
    prev = term;
}

// Compact case-sensitive vocabulary sidecar. The CT1 stream is front-coded
// alphabetically and has no per-term df because it is only a membership set.
static void put_case_terms(std::string &out,
                           const std::unordered_set<std::string> &case_terms) {
    std::vector<std::string> sorted(case_terms.begin(), case_terms.end());
    std::sort(sorted.begin(), sorted.end());
    out.append("CT1", 3);
    std::string prev;
    for (const auto &term : sorted) {
        size_t shared = 0;
        size_t mm = std::min(prev.size(), term.size());
        while (shared < mm && prev[shared] == term[shared]) shared++;
        put_vbyte(out, static_cast<long>(shared));
        put_vbyte(out, static_cast<long>(term.size() - shared));
        out.append(term, shared, term.size() - shared);
        prev = term;
    }
}

template <typename T>
static void wr(std::ofstream &f, T v) { f.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
template <typename T>
static T rd(std::ifstream &f) { T v; f.read(reinterpret_cast<char *>(&v), sizeof(T)); return v; }

// Elias-Fano encoder for the final postings.bin. One posting
// list of m = ords.size() ascending doc ordinals in [0,N): l = floor(log2(N/m))
// low bits/value packed LSB-first byte-padded; a unary high stream ((hi-prev)
// zeros then a 1) byte-padded; then, unless all_tf_one is true, (tf-1) in
// 128-blocks (width byte + packed). tf1.bin records which terms omit that
// final section.
// Must stay byte-identical to indexer._encode_postings and _index_cpp
// NativeIndex::get_postings (which recomputes l from m,N and decodes this).
static void put_postings_ef(std::string &out, const std::vector<uint32_t> &ords,
                            const std::vector<uint32_t> &tfs, long N,
                            bool all_tf_one = false) {
    long m = static_cast<long>(ords.size());
    int l = 0;
    if (m > 0) { long q = N / m; if (q >= 1) { while ((q >> 1) != 0) { q >>= 1; l++; } } }
    // Low bits.
    {
        uint64_t buf = 0; int nb = 0;
        uint32_t mask = (l > 0) ? ((1u << l) - 1u) : 0u;
        for (long i = 0; i < m; i++) {
            buf |= static_cast<uint64_t>(ords[i] & mask) << nb; nb += l;
            while (nb >= 8) { out.push_back(static_cast<char>(buf & 0xFF)); buf >>= 8; nb -= 8; }
        }
        if (nb) out.push_back(static_cast<char>(buf & 0xFF));
    }
    // High bits.
    {
        uint64_t buf = 0; int nb = 0; uint32_t prev = 0;
        for (long i = 0; i < m; i++) {
            uint32_t hi = ords[i] >> l;
            nb += static_cast<int>(hi - prev);   // (hi-prev) zero bits
            while (nb >= 8) { out.push_back(static_cast<char>(buf & 0xFF)); buf >>= 8; nb -= 8; }
            buf |= static_cast<uint64_t>(1) << nb; nb += 1;  // terminating one bit
            while (nb >= 8) { out.push_back(static_cast<char>(buf & 0xFF)); buf >>= 8; nb -= 8; }
            prev = hi;
        }
        if (nb) out.push_back(static_cast<char>(buf & 0xFF));
    }
    if (all_tf_one) return;
    // tf-1 in blocks of 128.
    long done = 0;
    while (done < m) {
        long k = std::min(static_cast<long>(128), m - done);
        uint32_t mx = 0;
        for (long j = 0; j < k; j++) { uint32_t v = tfs[done + j] - 1; if (v > mx) mx = v; }
        int width = 0; { uint32_t t = mx; while (t) { t >>= 1; width++; } }
        out.push_back(static_cast<char>(width));
        if (width > 0) {
            uint64_t buf = 0; int nb = 0;
            uint32_t wmask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
            for (long j = 0; j < k; j++) {
                buf |= static_cast<uint64_t>((tfs[done + j] - 1) & wmask) << nb; nb += width;
                while (nb >= 8) { out.push_back(static_cast<char>(buf & 0xFF)); buf >>= 8; nb -= 8; }
            }
            if (nb) out.push_back(static_cast<char>(buf & 0xFF));
        }
        done += k;
    }
}

// Elias-Fano encode a strictly increasing sequence in [0, universe), without
// term frequencies. Used by early.bin for the sparse posting positions whose
// early flag is set.
static void put_monotone_ef(std::string &out, const std::vector<uint32_t> &vals,
                            long universe) {
    long m = static_cast<long>(vals.size());
    int l = 0;
    if (m > 0) {
        long q = universe / m;
        if (q >= 1) {
            while ((q >> 1) != 0) { q >>= 1; l++; }
        }
    }
    {
        uint64_t buf = 0;
        int nb = 0;
        uint32_t mask = (l > 0) ? ((1u << l) - 1u) : 0u;
        for (long i = 0; i < m; i++) {
            buf |= static_cast<uint64_t>(vals[i] & mask) << nb;
            nb += l;
            while (nb >= 8) {
                out.push_back(static_cast<char>(buf & 0xFF));
                buf >>= 8;
                nb -= 8;
            }
        }
        if (nb) out.push_back(static_cast<char>(buf & 0xFF));
    }
    {
        uint64_t buf = 0;
        int nb = 0;
        uint32_t prev = 0;
        for (long i = 0; i < m; i++) {
            uint32_t hi = vals[i] >> l;
            nb += static_cast<int>(hi - prev);
            while (nb >= 8) {
                out.push_back(static_cast<char>(buf & 0xFF));
                buf >>= 8;
                nb -= 8;
            }
            buf |= static_cast<uint64_t>(1) << nb;
            nb++;
            while (nb >= 8) {
                out.push_back(static_cast<char>(buf & 0xFF));
                buf >>= 8;
                nb -= 8;
            }
            prev = hi;
        }
        if (nb) out.push_back(static_cast<char>(buf & 0xFF));
    }
}

// Sparse early-posting stream:
//   "SEF1"
//   ceil(num_terms/8) presence bits (term has at least one early posting)
//   for each present term: VByte(early_df), EF(posting positions, universe=df)
//
// Encoding positions within the ordinary posting list avoids repeating doc
// ordinals. The sparse representation is smaller than a full aligned flag
// stream and lets the reader retain only matching doc ordinals in memory.
template <typename PerTerm>
static std::string encode_early_sparse(const std::vector<PerTerm> &data) {
    std::string out("SEF1", 4);
    size_t bitmap_pos = out.size();
    out.resize(bitmap_pos + (data.size() + 7) / 8, '\0');
    for (size_t ti = 0; ti < data.size(); ti++) {
        const auto &flags = data[ti].early;
        size_t m = 0;
        for (uint8_t flag : flags) m += flag != 0;
        if (m == 0) continue;
        out[bitmap_pos + ti / 8] = static_cast<char>(
            static_cast<unsigned char>(out[bitmap_pos + ti / 8]) |
            static_cast<unsigned char>(1u << (ti % 8)));
        put_vbyte(out, static_cast<long>(m));
        std::vector<uint32_t> positions;
        positions.reserve(m);
        for (size_t i = 0; i < flags.size(); i++)
            if (flags[i]) positions.push_back(static_cast<uint32_t>(i));
        put_monotone_ef(out, positions, static_cast<long>(flags.size()));
    }
    return out;
}

// A block on disk: records sorted by term, each
//   [u32 term_len][term bytes][u32 df]
//   [df x u32 ord][df x u32 tf][df x u8 early]
struct BlockReader {
    std::ifstream f;
    bool ok = false;
    std::string term;
    std::vector<uint32_t> ords, tfs;
    std::vector<uint8_t> early;

    explicit BlockReader(const std::string &path) : f(path, std::ios::binary) { advance(); }

    void advance() {
        uint32_t tl;
        if (!f.read(reinterpret_cast<char *>(&tl), sizeof(tl))) { ok = false; return; }
        term.resize(tl);
        f.read(&term[0], tl);
        uint32_t df = rd<uint32_t>(f);
        ords.resize(df);
        tfs.resize(df);
        early.resize(df);
        f.read(reinterpret_cast<char *>(ords.data()), df * sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(tfs.data()), df * sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(early.data()), df * sizeof(uint8_t));
        ok = true;
    }
};

struct TermPostings {
    std::vector<uint32_t> ords, tfs;
    std::vector<uint8_t> early;
};

struct SelectedTerms {
    std::vector<std::string> kept;
    std::unordered_set<std::string> pruned;
    std::unordered_set<std::string> case_sensitive;
};

static std::map<std::string, uint32_t>
scan_document_frequencies(const std::vector<std::string> &block_files) {
    std::map<std::string, uint32_t> frequencies;
    for (const auto &path : block_files) {
        std::ifstream file(path, std::ios::binary);
        uint32_t term_length;
        while (file.read(reinterpret_cast<char *>(&term_length),
                         sizeof(term_length))) {
            std::string term(term_length, '\0');
            file.read(&term[0], term_length);
            uint32_t df = rd<uint32_t>(file);
            frequencies[term] += df;
            file.seekg(static_cast<std::streamoff>(df) *
                           (2 * sizeof(uint32_t) + sizeof(uint8_t)),
                       std::ios::cur);
        }
    }
    return frequencies;
}

static SelectedTerms select_terms(
    const std::map<std::string, uint32_t> &frequencies,
    const std::unordered_map<std::string, std::string> &canonical,
    double df_ratio) {
    SelectedTerms selected;
    selected.kept.reserve(frequencies.size());

    for (const auto &entry : frequencies) {
        const std::string &term = entry.first;
        bool keep = true;
        if (has_upper(term)) {
            uint32_t canonical_df = 0;
            auto canonical_it = canonical.find(term);
            if (canonical_it != canonical.end()) {
                auto frequency_it = frequencies.find(canonical_it->second);
                if (frequency_it != frequencies.end())
                    canonical_df = frequency_it->second;
            }
            keep = canonical_df == 0 ||
                   static_cast<double>(entry.second) / canonical_df <= df_ratio;
            if (keep)
                selected.case_sensitive.insert(term);
            else
                selected.pruned.insert(term);
        }
        if (keep) selected.kept.push_back(term);
    }
    return selected;
}

static std::vector<TermPostings> merge_blocks(
    const std::vector<std::string> &block_files,
    const std::vector<std::string> &kept_terms,
    const std::unordered_set<std::string> &pruned,
    const std::map<std::string, uint32_t> &frequencies) {
    std::unordered_map<std::string, size_t> term_slot;
    term_slot.reserve(kept_terms.size() * 2);
    for (size_t i = 0; i < kept_terms.size(); i++)
        term_slot[kept_terms[i]] = i;

    std::vector<BlockReader> readers;
    readers.reserve(block_files.size());
    for (const auto &path : block_files) readers.emplace_back(path);

    std::vector<TermPostings> data(kept_terms.size());
    while (true) {
        std::string next_term;
        bool found = false;
        for (const auto &reader : readers) {
            if (reader.ok && (!found || reader.term < next_term)) {
                next_term = reader.term;
                found = true;
            }
        }
        if (!found) break;

        const bool discard = pruned.count(next_term) != 0;
        TermPostings *postings = nullptr;
        if (!discard) {
            postings = &data[term_slot.at(next_term)];
            // The preceding DF scan gives the final size across every block,
            // so each output vector can allocate once instead of repeatedly
            // growing as worker/block fragments are appended.
            const size_t df = frequencies.at(next_term);
            postings->ords.reserve(df);
            postings->tfs.reserve(df);
            postings->early.reserve(df);
        }

        for (auto &reader : readers) {
            if (!reader.ok || reader.term != next_term) continue;
            // Pruned case variants still have to be consumed from each input
            // block, but never copy them into temporary/final merge vectors.
            if (!discard) {
                postings->ords.insert(
                    postings->ords.end(), reader.ords.begin(), reader.ords.end());
                postings->tfs.insert(
                    postings->tfs.end(), reader.tfs.begin(), reader.tfs.end());
                postings->early.insert(
                    postings->early.end(), reader.early.begin(), reader.early.end());
            }
            reader.advance();
        }
    }
    return data;
}

static void write_index(
    const std::string &dir,
    const std::vector<std::string> &terms,
    const std::vector<TermPostings> &data,
    const std::unordered_set<std::string> &case_terms,
    uint32_t document_count,
    long total_document_length) {
    std::string terms_blob;
    std::string postings_blob;
    std::string tf_one("TF1", 3);
    tf_one.resize(3 + (terms.size() + 7) / 8, '\0');

    std::string previous_term;
    for (size_t i = 0; i < terms.size(); i++) {
        put_term_frontcoded(terms_blob, previous_term, terms[i],
                            static_cast<long>(data[i].ords.size()));
        bool all_one = std::all_of(
            data[i].tfs.begin(), data[i].tfs.end(),
            [](uint32_t tf) { return tf == 1; });
        if (all_one)
            tf_one[3 + i / 8] = static_cast<char>(
                static_cast<unsigned char>(tf_one[3 + i / 8]) |
                static_cast<unsigned char>(1u << (i % 8)));
        put_postings_ef(postings_blob, data[i].ords, data[i].tfs,
                        static_cast<long>(document_count), all_one);
    }

    std::string early_blob = encode_early_sparse(data);
    std::ofstream(dir + "/terms.txt", std::ios::binary)
        .write(terms_blob.data(), terms_blob.size());
    std::ofstream(dir + "/postings.bin", std::ios::binary)
        .write(postings_blob.data(), postings_blob.size());
    std::ofstream(dir + "/tf1.bin", std::ios::binary)
        .write(tf_one.data(), tf_one.size());
    std::ofstream(dir + "/early.bin", std::ios::binary)
        .write(early_blob.data(), early_blob.size());

    std::string case_blob;
    put_case_terms(case_blob, case_terms);
    std::ofstream(dir + "/case_terms.bin", std::ios::binary)
        .write(case_blob.data(), case_blob.size());

    double average_length = document_count
        ? static_cast<double>(total_document_length) / document_count
        : 0.0;
    std::ofstream meta(dir + "/meta.json");
    meta << std::setprecision(std::numeric_limits<double>::max_digits10);
    meta << "{\"N\": " << document_count
         << ", \"avg_doc_len\": " << average_length << "}";
    meta.close();
    compact_docids(dir);
}

struct SpimiBuilder {
    std::string dir;
    size_t flush_threshold;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> block;
    std::unordered_map<std::string, std::vector<uint8_t>> block_early;
    size_t cur_postings = 0;
    uint32_t n_docs = 0;      // next doc gets this ordinal (starts at start_ord)
    uint32_t local_docs = 0;  // number of add_document() calls on THIS instance
    long total_len = 0;
    std::vector<std::string> block_files;
    std::ofstream docs_out;
    // During the build, temporary doc-ID lines and the parallel VByte
    // doclen.bin stream are written in lockstep. Finalization may replace
    // docs.txt with the compact docids.bin representation.
    std::ofstream doclen_out;
    std::unordered_set<std::string> case_terms_seen;

    // Fused tokenizer (owns the stopword set + stem cache for this worker) and a
    // reusable token buffer, so add_document_from_text allocates no Python list.
    Tokenizer tok;
    std::vector<std::string> tok_buf;
    std::vector<std::string> early_buf;

    SpimiBuilder(const std::string &d, size_t threshold, uint32_t start_ord = 0)
        : dir(d), flush_threshold(threshold), n_docs(start_ord), docs_out(d + "/docs.txt"),
          doclen_out(d + "/doclen.bin", std::ios::binary) {}

    void add_document(const std::string &doc_id, const std::vector<std::string> &tokens, int doc_len) {
        docs_out << doc_id << '\n';
        {
            std::string lenb;
            put_vbyte(lenb, doc_len);
            doclen_out.write(lenb.data(), lenb.size());
        }
        uint32_t ord = n_docs++;
        local_docs++;
        total_len += doc_len;

        std::unordered_map<std::string, uint32_t> tf;
        std::unordered_set<std::string> early_terms;
        for (const auto &t : tokens) {
            if (is_early_term(t)) {
                early_terms.insert(t.substr(EARLY_PREFIX.size()));
                continue;
            }
            tf[t]++;
            if (::has_upper(t)) case_terms_seen.insert(t);
        }
        for (auto &kv : tf) {
            auto &vec = block[kv.first];
            vec.push_back({ord, kv.second});
            block_early[kv.first].push_back(
                static_cast<uint8_t>(early_terms.count(kv.first) != 0));
            cur_postings++;
        }

        if (cur_postings >= flush_threshold) flush_block();
    }

    // Configure the fused tokenizer's stopword filter (already-stemmed stems).
    void set_stopword_stems(const std::vector<std::string> &stems) {
        tok.set_stopword_stems(stems);
    }

    // Fused build entry point: tokenise+stem `text` in C++ and index it, without
    // ever building a Python token list. Byte-identical to
    // add_document(doc_id, tokenize_doc(text)...).
    void add_document_from_text(const std::string &doc_id, const std::string &text) {
        int doc_len = tok.tokenize(text, tok_buf);
        // Tokenize the first 12 whitespace-delimited raw words with the same
        // tokenizer. Internal marker tokens let add_document() set a
        // transient early-prefix flag; the markers never enter the persisted
        // vocabulary. finalize() writes the surviving flags sparsely to
        // early.bin.
        std::string prefix;
        size_t pos = 0;
        int words = 0;
        while (pos < text.size() && words < 12) {
            while (pos < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
            if (pos >= text.size()) break;
            size_t begin = pos;
            while (pos < text.size() &&
                   !std::isspace(static_cast<unsigned char>(text[pos]))) pos++;
            if (!prefix.empty()) prefix.push_back(' ');
            prefix.append(text, begin, pos - begin);
            words++;
        }
        tok.tokenize(prefix, early_buf);
        for (const auto &t : early_buf) tok_buf.push_back(EARLY_PREFIX + t);
        add_document(doc_id, tok_buf, doc_len);
    }

    void flush_block() {
        if (block.empty()) return;
        std::vector<std::string> terms;
        terms.reserve(block.size());
        for (auto &kv : block) terms.push_back(kv.first);
        std::sort(terms.begin(), terms.end());

        std::string path = dir + "/block_" + std::to_string(block_files.size()) + ".tmp";
        std::ofstream f(path, std::ios::binary);
        for (auto &term : terms) {
            auto &vec = block[term];  // already ascending in ord (docs added in order)
            uint32_t tl = static_cast<uint32_t>(term.size());
            uint32_t df = static_cast<uint32_t>(vec.size());
            wr(f, tl);
            f.write(term.data(), tl);
            wr(f, df);
            for (auto &p : vec) wr(f, p.first);
            for (auto &p : vec) wr(f, p.second);
            auto &early = block_early[term];
            f.write(reinterpret_cast<const char *>(early.data()),
                    static_cast<std::streamsize>(early.size()));
        }
        block.clear();
        block_early.clear();
        cur_postings = 0;
        block_files.push_back(path);
    }

    std::vector<std::string> case_terms() {
        return std::vector<std::string>(case_terms_seen.begin(), case_terms_seen.end());
    }

    // --- Parallel-build worker API ---------------------------------------
    // Used when a SpimiBuilder is one of N per-worker builders (each owning
    // a contiguous, globally-ordinalled slice of the corpus) instead of the
    // sole builder for the whole corpus. close_worker() flushes the last
    // in-memory block and closes this worker's document sidecars
    // without doing any cross-worker merge work; the parent process then
    // hands the accessors below to finalize_parallel() to do the k-way
    // merge (see finalize_parallel below for how ord/df-hash-order identity
    // with the serial path is preserved).
    void close_worker() {
        flush_block();
        docs_out.close();
        doclen_out.close();
    }

    std::vector<std::string> get_block_files() const { return block_files; }
    uint32_t get_local_docs() const { return local_docs; }
    long get_total_len() const { return total_len; }
    std::string docs_path() const { return dir + "/docs.txt"; }
    std::string doclen_path() const { return dir + "/doclen.bin"; }

    void finalize(const std::unordered_map<std::string, std::string> &canonical, double df_ratio) {
        flush_block();
        docs_out.close();
        doclen_out.close();

        auto frequencies = scan_document_frequencies(block_files);
        auto selected = select_terms(frequencies, canonical, df_ratio);
        auto data = merge_blocks(
            block_files, selected.kept, selected.pruned, frequencies);
        write_index(
            dir, selected.kept, data, selected.case_sensitive, n_docs, total_len);

        for (auto &path : block_files) std::remove(path.c_str());
    }
};

// --- Parallel build: merge N independent per-worker SpimiBuilders --------
//
// Each worker owns a contiguous, globally-ordinalled slice of the corpus
// (worker i's docs already carry final global ordinals — see the start_ord
// constructor arg — so postings never need an ordinal remap here). This
// function reproduces exactly what a single serial SpimiBuilder::finalize()
// would have produced, byte for byte, which requires two non-obvious
// matches to the serial code path, not just the same final content:
//
// 1. Pass-2 (terms.txt/postings.bin) ordinal order: for a given term, ords
//    across workers must be concatenated in ascending order. Because each
//    worker's ordinal range is contiguous and workers are processed in
//    index order (0, 1, 2, ...), and each worker's own blocks are already
//    internally ascending (docs added in order), simply building the
//    k-way-merge `readers` list in [worker0's blocks..., worker1's
//    blocks..., ...] order guarantees this — no ordinal remap needed.
//
// 2. Document frequencies and truecase handling use the same shared helpers
//    as serial finalization, so both paths apply identical filtering.
//    put_case_terms() sorts the retained case-sensitive terms before
//    writing case_terms.bin, so unordered_set iteration order no
//    longer affects the serialized bytes.
struct WorkerData {
    std::vector<std::string> block_files;
    std::string docs_txt;
    std::string doclen_bin;
    uint32_t n_docs;  // LOCAL doc count for this worker (not a global ordinal).
    long total_len;
};

static void finalize_worker_data(
    const std::vector<WorkerData> &ws, const std::string &out_dir,
    const std::unordered_map<std::string, std::string> &canonical,
    double df_ratio) {
    // Concatenate docs.txt / doclen.bin shards in worker order == global
    // doc-ordinal order (A4: doc-id lines and VByte lengths are now two
    // separate parallel streams; both concatenate the same way).
    {
        std::ofstream docs_out(out_dir + "/docs.txt", std::ios::binary);
        for (auto &w : ws) {
            std::ifstream in(w.docs_txt, std::ios::binary);
            docs_out << in.rdbuf();
        }
    }
    {
        std::ofstream doclen_out(out_dir + "/doclen.bin", std::ios::binary);
        for (auto &w : ws) {
            std::ifstream in(w.doclen_bin, std::ios::binary);
            doclen_out << in.rdbuf();
        }
    }

    uint32_t N = 0;
    long total_len = 0;
    for (auto &w : ws) { N += w.n_docs; total_len += w.total_len; }

    std::vector<std::string> all_blocks;
    for (auto &w : ws)
        for (auto &p : w.block_files) all_blocks.push_back(p);

    auto frequencies = scan_document_frequencies(all_blocks);
    auto selected = select_terms(frequencies, canonical, df_ratio);
    auto data = merge_blocks(
        all_blocks, selected.kept, selected.pruned, frequencies);
    write_index(
        out_dir, selected.kept, data, selected.case_sensitive, N, total_len);

    for (auto &path : all_blocks) std::remove(path.c_str());
    for (auto &w : ws) {
        std::remove(w.docs_txt.c_str());    // copied into the final temp stream above
        std::remove(w.doclen_bin.c_str());  // copied into the final temp stream above
    }
}

void finalize_parallel(const py::list &workers, const std::string &out_dir,
                       const std::unordered_map<std::string, std::string> &canonical,
                       double df_ratio) {
    std::vector<WorkerData> ws;
    for (auto &item : workers) {
        py::dict d = item.cast<py::dict>();
        WorkerData w;
        w.block_files = d["block_files"].cast<std::vector<std::string>>();
        w.docs_txt = d["docs_txt"].cast<std::string>();
        w.doclen_bin = d["doclen_bin"].cast<std::string>();
        w.n_docs = d["n_docs"].cast<uint32_t>();
        w.total_len = d["total_len"].cast<long>();
        ws.push_back(std::move(w));
    }
    finalize_worker_data(ws, out_dir, canonical, df_ratio);
}

void build_index_from_jsonl(
    const std::string &corpus_path, const std::string &index_dir,
    const std::vector<std::string> &stopword_stems,
    size_t worker_flush_threshold, int requested_workers, double df_ratio) {
    std::ifstream scan(corpus_path, std::ios::binary);
    if (!scan)
        throw std::runtime_error("cannot open corpus: " + corpus_path);

    std::vector<uint64_t> offsets;
    std::string line;
    while (true) {
        std::streampos position = scan.tellg();
        if (!std::getline(scan, line)) break;
        if (!is_blank_line(line))
            offsets.push_back(static_cast<uint64_t>(position));
    }

    if (offsets.empty()) {
        finalize_worker_data({}, index_dir, {}, df_ratio);
        return;
    }

    int worker_count = std::min<int>(
        std::max(1, requested_workers), static_cast<int>(offsets.size()));
    std::vector<WorkerData> workers(static_cast<size_t>(worker_count));
    std::vector<std::unordered_set<std::string>> worker_case_terms(
        static_cast<size_t>(worker_count));
    std::vector<std::exception_ptr> errors(static_cast<size_t>(worker_count));
    std::vector<std::string> worker_dirs(static_cast<size_t>(worker_count));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(worker_count));

    size_t base = offsets.size() / static_cast<size_t>(worker_count);
    size_t remainder = offsets.size() % static_cast<size_t>(worker_count);
    size_t start = 0;
    for (int worker_id = 0; worker_id < worker_count; worker_id++) {
        size_t count = base + (
            static_cast<size_t>(worker_id) < remainder ? 1u : 0u);
        size_t worker_start = start;
        uint32_t start_ordinal = static_cast<uint32_t>(start);
        std::string worker_dir =
            index_dir + "/_w" + std::to_string(worker_id);
        worker_dirs[static_cast<size_t>(worker_id)] = worker_dir;
        start += count;

        threads.emplace_back([&, worker_id, count, worker_start,
                              start_ordinal, worker_dir] {
            try {
                remove_directory_tree(worker_dir);
                ensure_directory(worker_dir);
                SpimiBuilder builder(
                    worker_dir, std::max<size_t>(1, worker_flush_threshold),
                    start_ordinal);
                builder.set_stopword_stems(stopword_stems);

                std::ifstream input(corpus_path, std::ios::binary);
                if (!input)
                    throw std::runtime_error("cannot open corpus: " + corpus_path);
                input.seekg(static_cast<std::streamoff>(offsets[worker_start]));
                if (!input)
                    throw std::runtime_error("cannot seek corpus: " + corpus_path);

                size_t documents_read = 0;
                std::string worker_line;
                while (documents_read < count &&
                       std::getline(input, worker_line)) {
                    if (is_blank_line(worker_line)) continue;
                    try {
                        auto record = parse_corpus_record(worker_line);
                        builder.add_document_from_text(
                            record.first, record.second);
                    } catch (const std::exception &error) {
                        throw std::runtime_error(
                            "corpus record " +
                            std::to_string(worker_start + documents_read + 1) +
                            ": " + error.what());
                    }
                    documents_read++;
                }
                if (documents_read != count)
                    throw std::runtime_error(
                        "corpus ended before assigned worker range");

                builder.close_worker();
                WorkerData result;
                result.block_files = builder.get_block_files();
                result.docs_txt = builder.docs_path();
                result.doclen_bin = builder.doclen_path();
                result.n_docs = builder.get_local_docs();
                result.total_len = builder.get_total_len();
                workers[static_cast<size_t>(worker_id)] = std::move(result);
                auto terms = builder.case_terms();
                worker_case_terms[static_cast<size_t>(worker_id)].insert(
                    terms.begin(), terms.end());
            } catch (...) {
                errors[static_cast<size_t>(worker_id)] =
                    std::current_exception();
            }
        });
    }

    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) {
        if (error) {
            for (const auto &worker_dir : worker_dirs)
                remove_directory_tree(worker_dir);
            std::rethrow_exception(error);
        }
    }

    std::unordered_map<std::string, std::string> canonical;
    for (const auto &case_terms : worker_case_terms) {
        for (const auto &term : case_terms)
            canonical[term] = Tokenizer::stem(term);
    }

    try {
        finalize_worker_data(workers, index_dir, canonical, df_ratio);
    } catch (...) {
        for (const auto &worker_dir : worker_dirs)
            remove_directory_tree(worker_dir);
        throw;
    }
    for (const auto &worker_dir : worker_dirs)
        ::rmdir(worker_dir.c_str());
}

PYBIND11_MODULE(_spimi_cpp, m) {
    m.doc() = "SPIMI index builder: block flushing and compact v2 serialization.";
    py::class_<SpimiBuilder>(m, "SpimiBuilder")
        .def(py::init<const std::string &, size_t, uint32_t>(),
             py::arg("index_dir"), py::arg("flush_threshold"), py::arg("start_ord") = 0)
        .def("set_stopword_stems", &SpimiBuilder::set_stopword_stems,
             py::arg("stems"))
        .def("add_document_from_text", &SpimiBuilder::add_document_from_text,
             py::arg("doc_id"), py::arg("text"))
        .def("add_document", &SpimiBuilder::add_document,
             py::arg("doc_id"), py::arg("tokens"), py::arg("doc_len"))
        .def("case_terms", &SpimiBuilder::case_terms)
        .def("finalize", &SpimiBuilder::finalize, py::arg("canonical"), py::arg("df_ratio"))
        .def("close_worker", &SpimiBuilder::close_worker)
        .def("block_files", &SpimiBuilder::get_block_files)
        .def("n_docs_count", &SpimiBuilder::get_local_docs)
        .def("total_len_count", &SpimiBuilder::get_total_len)
        .def("docs_path", &SpimiBuilder::docs_path)
        .def("doclen_path", &SpimiBuilder::doclen_path);
    m.def("finalize_parallel", &finalize_parallel,
          py::arg("workers"), py::arg("out_dir"), py::arg("canonical"), py::arg("df_ratio"));
    m.def("build_index_from_jsonl", &build_index_from_jsonl,
          py::arg("corpus_path"), py::arg("index_dir"),
          py::arg("stopword_stems"), py::arg("worker_flush_threshold"),
          py::arg("workers"), py::arg("df_ratio"),
          py::call_guard<py::gil_scoped_release>());
}
