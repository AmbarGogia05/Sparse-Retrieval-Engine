// submission/_index_cpp.cpp — optional native index/query accelerator.
//
// Loads the compact v2 index (terms.txt, postings.bin, doclen.bin, and the
// optional docids.bin, tf1.bin, early.bin, and case_terms.bin sidecars) into
// contiguous C++ arrays. It owns native BM25/VSM scoring, coverage and early
// matching, RM3 support, and reconstruction of the forward view needed by RM3.
//
// The primary retrieve() binding accepts the raw query and performs
// tokenisation, scoring, RRF fusion, and final reranking entirely in C++, so a
// query and its final top-k cross the Python/native boundary only once each.
// The component methods remain exposed for tuning scripts and diagnostics.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <thread>

#include "tokenizer.h"

namespace py = pybind11;

struct NativeIndex {
    using OrdHit = std::pair<int, double>;
    using DocHit = std::pair<std::string, double>;

    int N = 0;
    double avg_doc_len = 0.0;
    std::vector<std::string> docid;   // ordinal -> doc_id string
    std::vector<int> doc_len;         // ordinal -> length
    std::vector<double> doc_norm;     // ordinal -> sqrt(VSM norm)
    std::unordered_map<std::string, int> term_id;
    std::vector<std::vector<int>> t_ords;  // term -> sorted doc ordinals
    std::vector<std::vector<int>> t_tfs;   // term -> term frequencies
    std::vector<std::vector<int>> t_early_ords; // term -> docs with prefix presence
    std::vector<int> t_df;                 // term -> document frequency

    // Forward index (CSR): for RM3, doc d's terms are fwd_terms[fwd_off[d] ..
    // fwd_off[d+1]) with parallel fwd_tfs. Reconstructed at load as the
    // transpose of the inverted postings (see load()) — not persisted to disk.
    bool has_forward = false;
    std::vector<size_t> fwd_off;
    std::vector<int> fwd_terms;
    std::vector<int> fwd_tfs;
    Tokenizer query_tok;

    explicit NativeIndex(
        const std::string &dir,
        const std::vector<std::string> &stopword_stems = {}) {
        query_tok.set_stopword_stems(stopword_stems);
        load(dir);
    }

    static std::string read_binary(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        return {
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        };
    }

    static long vbyte(const unsigned char *p, size_t &pos) {
        long n = 0;
        int s = 0;
        unsigned char b;
        do {
            b = p[pos++];
            n |= static_cast<long>(b & 0x7F) << s;
            s += 7;
        } while (!(b & 0x80));
        return n;
    }

    // floor(log2(N/m)) == bit_length(N/m)-1; the Elias-Fano low-bit width.
    static int ef_l(long m, long N) {
        if (m <= 0) return 0;
        long q = N / m;
        if (q < 1) return 0;
        int l = 0;
        while ((q >> 1) != 0) { q >>= 1; l++; }
        return l;
    }

    struct PackedBitReader {
        const unsigned char *p;
        size_t pos;
        uint64_t buffer = 0;
        int bits = 0;

        PackedBitReader(const unsigned char *data, size_t start)
            : p(data), pos(start) {}

        uint32_t read(int width) {
            if (width == 0) return 0;
            while (bits < width) {
                buffer |= static_cast<uint64_t>(p[pos++]) << bits;
                bits += 8;
            }
            uint64_t mask = width == 32
                ? 0xFFFFFFFFULL
                : ((1ULL << width) - 1ULL);
            uint32_t value = static_cast<uint32_t>(buffer & mask);
            buffer >>= width;
            bits -= width;
            return value;
        }

        uint32_t read_vbyte() {
            uint32_t value = 0;
            int shift = 0;
            while (true) {
                uint32_t byte = read(8);
                value |= (byte & 0x7Fu) << shift;
                if (byte & 0x80u) return value;
                shift += 7;
            }
        }
    };

    // TF2-TF6 decoder. TF5+ keep the bitstream continuous across terms;
    // earlier packed versions byte-pad each term.
    static void get_postings_packed(
        PackedBitReader &reader, long df, long N,
        std::vector<int> &ords, std::vector<int> &tfs,
        bool all_tf_one = false, bool gap_rice = false,
        bool sparse_tf = false, bool adaptive_gap_rice = false) {
        long m = df;
        int l = ef_l(m, N);
        ords.resize(static_cast<size_t>(m));
        if (gap_rice) {
            int64_t previous = -1;
            for (long start = 0; start < m; start += 32) {
                long count = std::min(32L, m - start);
                int rice_k = l;
                if (adaptive_gap_rice && reader.read(1)) {
                    int code = static_cast<int>(reader.read(3));
                    rice_k = code == 7
                        ? static_cast<int>(reader.read(5))
                        : l + code - 3;
                }
                for (long j = 0; j < count; j++) {
                    uint32_t quotient = 0;
                    while (reader.read(1) != 0) quotient++;
                    uint32_t remainder = reader.read(rice_k);
                    previous += static_cast<int64_t>(
                        ((quotient << rice_k) | remainder) + 1u);
                    ords[static_cast<size_t>(start + j)] =
                        static_cast<int>(previous);
                }
            }
        } else {
            std::vector<uint32_t> lows(static_cast<size_t>(m), 0);
            for (long i = 0; i < m; i++)
                lows[static_cast<size_t>(i)] = reader.read(l);

            long high = 0;
            long ones = 0;
            while (ones < m) {
                if (reader.read(1)) {
                    ords[static_cast<size_t>(ones)] = static_cast<int>(
                        (high << l) | lows[static_cast<size_t>(ones)]);
                    ones++;
                } else {
                    high++;
                }
            }
        }

        tfs.resize(static_cast<size_t>(m));
        if (all_tf_one) {
            std::fill(tfs.begin(), tfs.end(), 1);
            return;
        }

        auto read_tf_values = [&](long count) {
            std::vector<uint32_t> values(static_cast<size_t>(count));
            long done = 0;
            while (done < count) {
                long block_size =
                    std::min(static_cast<long>(128), count - done);
                int header = static_cast<int>(reader.read(8));
                if (header & 0x80) {
                    int rice_k = header & 0x1F;
                    for (long j = 0; j < block_size; j++) {
                        uint32_t quotient = 0;
                        while (reader.read(1) != 0) quotient++;
                        uint32_t remainder = reader.read(rice_k);
                        values[static_cast<size_t>(done + j)] =
                            (quotient << rice_k) | remainder;
                    }
                } else {
                    int width = header;
                    for (long j = 0; j < block_size; j++)
                        values[static_cast<size_t>(done + j)] =
                            reader.read(width);
                }
                done += block_size;
            }
            return values;
        };

        if (sparse_tf) {
            std::fill(tfs.begin(), tfs.end(), 1);
            long exception_count =
                static_cast<long>(reader.read_vbyte());
            int position_k = ef_l(exception_count, m);
            std::vector<int> positions(
                static_cast<size_t>(exception_count));
            int64_t previous = -1;
            for (long i = 0; i < exception_count; i++) {
                uint32_t quotient = 0;
                while (reader.read(1) != 0) quotient++;
                uint32_t remainder = reader.read(position_k);
                previous += static_cast<int64_t>(
                    ((quotient << position_k) | remainder) + 1u);
                positions[static_cast<size_t>(i)] =
                    static_cast<int>(previous);
            }
            auto excess = read_tf_values(exception_count);
            for (long i = 0; i < exception_count; i++)
                tfs[static_cast<size_t>(
                    positions[static_cast<size_t>(i)])] =
                    static_cast<int>(
                        excess[static_cast<size_t>(i)] + 2u);
        } else {
            auto values = read_tf_values(m);
            for (long i = 0; i < m; i++)
                tfs[static_cast<size_t>(i)] =
                    static_cast<int>(
                        values[static_cast<size_t>(i)] + 1u);
        }
    }

    static void get_postings_packed_term(
        const unsigned char *p, size_t &pos, long df, long N,
        std::vector<int> &ords, std::vector<int> &tfs,
        bool all_tf_one = false, bool gap_rice = false,
        bool sparse_tf = false) {
        PackedBitReader reader(p, pos);
        get_postings_packed(
            reader, df, N, ords, tfs, all_tf_one, gap_rice, sparse_tf);
        pos = reader.pos;
    }

    // Inverse of _spimi_cpp.cpp's put_postings_ef / indexer._encode_postings:
    // legacy TF1 Elias-Fano streams, whose low bits, high bits, and individual
    // TF blocks are each byte-aligned.
    static void get_postings(const unsigned char *p, size_t &pos, long df, long N,
                             std::vector<int> &ords, std::vector<int> &tfs,
                             bool all_tf_one = false) {
        long m = df;
        get_monotone_ef(p, pos, m, N, ords);
        tfs.resize(m);
        if (all_tf_one) {
            tfs.assign(m, 1);
            return;
        }
        // tf blocks: (tf-1) in blocks of 128, each independently fixed-width
        // or Rice-coded according to its header.
        long done = 0;
        while (done < m) {
            long block_size = std::min(static_cast<long>(128), m - done);
            int header = p[pos++];
            if (header & 0x80) {
                int rice_k = header & 0x1F;
                uint64_t buf = 0;
                int nb = 0;
                auto read_bits = [&](int bits) -> uint32_t {
                    while (nb < bits) {
                        buf |= static_cast<uint64_t>(p[pos++]) << nb;
                        nb += 8;
                    }
                    uint32_t value = bits
                        ? static_cast<uint32_t>(
                              buf & ((1ULL << bits) - 1ULL))
                        : 0u;
                    buf >>= bits;
                    nb -= bits;
                    return value;
                };
                for (long j = 0; j < block_size; j++) {
                    uint32_t quotient = 0;
                    while (read_bits(1) != 0) quotient++;
                    uint32_t remainder =
                        rice_k ? read_bits(rice_k) : 0u;
                    uint32_t value =
                        (quotient << rice_k) | remainder;
                    tfs[done + j] = static_cast<int>(value + 1u);
                }
            } else {
                int width = header;
                if (width == 0) {
                    for (long j = 0; j < block_size; j++)
                        tfs[done + j] = 1;
                } else {
                    uint64_t buf = 0;
                    int nb = 0;
                    uint64_t wmask = (1ULL << width) - 1ULL;
                    for (long j = 0; j < block_size; j++) {
                        while (nb < width) {
                            buf |= static_cast<uint64_t>(p[pos++]) << nb;
                            nb += 8;
                        }
                        tfs[done + j] =
                            static_cast<int>((buf & wmask) + 1u);
                        buf >>= width;
                        nb -= width;
                    }
                }
            }
            done += block_size;
        }
    }

    // Decode the no-TF Elias-Fano stream used by sparse early.bin records.
    static void get_monotone_ef(const unsigned char *p, size_t &pos, long m,
                                long universe, std::vector<int> &vals) {
        vals.resize(m);
        int l = ef_l(m, universe);
        std::vector<int> lows(m, 0);
        if (l > 0) {
            uint64_t buf = 0;
            int nb = 0;
            size_t bi = pos;
            long mask = (1L << l) - 1;
            for (long i = 0; i < m; i++) {
                while (nb < l) {
                    buf |= static_cast<uint64_t>(p[bi++]) << nb;
                    nb += 8;
                }
                lows[i] = static_cast<int>(buf & mask);
                buf >>= l;
                nb -= l;
            }
        }
        pos += static_cast<size_t>((m * static_cast<long>(l) + 7) / 8);
        uint64_t buf = 0;
        int nb = 0;
        size_t bi = pos;
        long cur = 0;
        long ones = 0;
        while (ones < m) {
            if (nb == 0) {
                buf = p[bi++];
                nb = 8;
            }
            int bit = static_cast<int>(buf & 1);
            buf >>= 1;
            nb--;
            if (bit) {
                vals[ones] = static_cast<int>(
                    (cur << l) | static_cast<long>(lows[ones]));
                ones++;
            } else {
                cur++;
            }
        }
        pos = bi;
    }

    void load_documents(const std::string &dir) {
        // Document lengths are stored in the parallel VByte stream doclen.bin.
        // Prefer the compact D36 docids.bin sidecar when all IDs fit its
        // 8-character base-36 representation; otherwise use docs.txt.
        {
            std::ifstream packed(dir + "/docids.bin", std::ios::binary);
            char magic[4] = {};
            if (packed.read(magic, 4) && std::string(magic, 3) == "D36") {
                int version = static_cast<unsigned char>(magic[3]);
                packed.seekg(0, std::ios::end);
                std::streamoff total = packed.tellg();
                packed.seekg(4, std::ios::beg);
                uint64_t count = 0;
                if (version == 1)
                    count = total >= 4
                        ? static_cast<uint64_t>((total - 4) / 6)
                        : 0;
                else if (version == 2)
                    count = total >= 4
                        ? static_cast<uint64_t>((total - 4) * 8 / 42)
                        : 0;

                docid.reserve(static_cast<size_t>(count));
                uint64_t buffer = 0;
                int bits = 0;
                for (uint64_t i = 0; i < count; i++) {
                    uint64_t value;
                    if (version == 1) {
                        value = 0;
                        for (int j = 0; j < 6; j++)
                            value |= static_cast<uint64_t>(
                                static_cast<unsigned char>(packed.get()))
                                << (8 * j);
                    } else {
                        while (bits < 42) {
                            buffer |= static_cast<uint64_t>(
                                static_cast<unsigned char>(packed.get()))
                                << bits;
                            bits += 8;
                        }
                        value = buffer & ((1ULL << 42) - 1ULL);
                        buffer >>= 42;
                        bits -= 42;
                    }
                    std::string id(8, '0');
                    for (int j = 7; j >= 0; j--) {
                        int digit = static_cast<int>(value % 36u);
                        id[j] = digit < 10 ? static_cast<char>('0' + digit)
                                           : static_cast<char>('a' + digit - 10);
                        value /= 36u;
                    }
                    docid.push_back(std::move(id));
                }
                if (version != 1 && version != 2) docid.clear();
            }
            if (docid.empty()) {
                std::ifstream f(dir + "/docs.txt");
                std::string line;
                while (std::getline(f, line)) docid.push_back(line);
            }
        }
        N = static_cast<int>(docid.size());
        {
            std::string db = read_binary(dir + "/doclen.bin");
            const unsigned char *dp = reinterpret_cast<const unsigned char *>(db.data());
            size_t dpos = 0;
            doc_len.reserve(N);
            for (int d = 0; d < N; d++) doc_len.push_back(static_cast<int>(vbyte(dp, dpos)));
        }
        long tot = 0;
        for (int L : doc_len) tot += L;
        avg_doc_len = static_cast<double>(tot) / N;
    }

    void load_postings(const std::string &dir) {
        std::string blob = read_binary(dir + "/postings.bin");
        const unsigned char *p = reinterpret_cast<const unsigned char *>(blob.data());
        size_t pos = 0;

        // TM2 separates NUL-terminated front-coded terms from the contiguous
        // VByte df stream for better compression locality. The original
        // interleaved format remains readable for existing indices.
        std::vector<std::string> terms;
        std::vector<int> dfs;
        {
            std::string tb = read_binary(dir + "/terms.txt");
            const unsigned char *tp = reinterpret_cast<const unsigned char *>(tb.data());
            size_t tpos = 0, tn = tb.size();
            std::string prev;
            if (tn >= 3 && tb.compare(0, 3, "TM2") == 0) {
                tpos = 3;
                long count = vbyte(tp, tpos);
                terms.reserve(static_cast<size_t>(count));
                dfs.reserve(static_cast<size_t>(count));
                for (long i = 0; i < count; i++) {
                    long shared = vbyte(tp, tpos);
                    size_t end = tpos;
                    while (end < tn && tp[end] != 0) end++;
                    std::string term =
                        prev.substr(0, static_cast<size_t>(shared));
                    term.append(
                        reinterpret_cast<const char *>(tp + tpos),
                        end - tpos);
                    tpos = end + 1;
                    terms.push_back(term);
                    prev = term;
                }
                for (long i = 0; i < count; i++)
                    dfs.push_back(static_cast<int>(vbyte(tp, tpos)));
            } else {
                while (tpos < tn) {
                    long shared = vbyte(tp, tpos);
                    long suflen = vbyte(tp, tpos);
                    std::string term =
                        prev.substr(0, static_cast<size_t>(shared));
                    term.append(
                        reinterpret_cast<const char *>(tp + tpos),
                        static_cast<size_t>(suflen));
                    tpos += static_cast<size_t>(suflen);
                    long df = vbyte(tp, tpos);
                    terms.push_back(term);
                    dfs.push_back(static_cast<int>(df));
                    prev = term;
                }
            }
        }
        int T = static_cast<int>(terms.size());
        t_ords.resize(T);
        t_tfs.resize(T);
        t_early_ords.resize(T);
        t_df.resize(T);
        std::vector<uint8_t> tf_mode(T, 0);
        bool packed_postings = false;
        bool gap_rice_postings = false;
        bool sparse_tf_postings = false;
        bool continuous_postings = false;
        bool adaptive_gap_postings = false;
        {
            std::string tfb = read_binary(dir + "/tf1.bin");
            size_t bytes = (static_cast<size_t>(T) + 7) / 8;
            adaptive_gap_postings =
                tfb.size() >= 3 && tfb.compare(0, 3, "TF6") == 0;
            continuous_postings =
                adaptive_gap_postings ||
                (tfb.size() >= 3 && tfb.compare(0, 3, "TF5") == 0);
            sparse_tf_postings =
                continuous_postings ||
                (tfb.size() >= 3 && tfb.compare(0, 3, "TF4") == 0);
            gap_rice_postings =
                sparse_tf_postings ||
                (tfb.size() >= 3 && tfb.compare(0, 3, "TF3") == 0);
            packed_postings =
                gap_rice_postings ||
                (tfb.size() >= 3 && tfb.compare(0, 3, "TF2") == 0);
            if (sparse_tf_postings && tfb.size() >= 3 + bytes) {
                for (int ti = 0; ti < T; ti++)
                    tf_mode[ti] =
                        ((static_cast<unsigned char>(tfb[3 + ti / 8])
                          >> (ti % 8)) & 1u)
                            ? 1u
                            : 2u;
            } else if (
                tfb.size() >= 3 + bytes &&
                (tfb.compare(0, 3, "TF1") == 0 || packed_postings)) {
                for (int ti = 0; ti < T; ti++)
                    tf_mode[ti] = static_cast<uint8_t>(
                        (static_cast<unsigned char>(tfb[3 + ti / 8])
                         >> (ti % 8)) & 1u);
            }
        }
        PackedBitReader continuous_reader(p, 0);
        for (int ti = 0; ti < T; ti++) {
            int df = dfs[ti];
            t_df[ti] = df;
            term_id[terms[ti]] = ti;
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            if (continuous_postings)
                get_postings_packed(
                    continuous_reader, df, N, ords, tfs,
                    tf_mode[ti] == 1, true, tf_mode[ti] == 2,
                    adaptive_gap_postings);
            else if (packed_postings)
                get_postings_packed_term(
                    p, pos, df, N, ords, tfs, tf_mode[ti] == 1,
                    gap_rice_postings, tf_mode[ti] == 2);
            else
                get_postings(
                    p, pos, df, N, ords, tfs, tf_mode[ti] == 1);
        }
    }

    void load_early_matches(const std::string &dir) {
        int T = static_cast<int>(t_df.size());
        // Optional prefix-presence stream. SEF2 stores Rice-coded gaps between
        // sparse posting positions; SEF1 used Elias-Fano. Both are accepted,
        // as is the original dense packed-bit stream.
        {
            std::string eb = read_binary(dir + "/early.bin");
            const unsigned char *ep =
                reinterpret_cast<const unsigned char *>(eb.data());
            bool sparse_rice =
                eb.size() >= 4 && eb.compare(0, 4, "SEF2") == 0;
            bool sparse_ef =
                eb.size() >= 4 && eb.compare(0, 4, "SEF1") == 0;
            if (sparse_rice || sparse_ef) {
                size_t bitmap_pos = 4;
                size_t bitmap_bytes = (static_cast<size_t>(T) + 7) / 8;
                size_t epos = bitmap_pos + bitmap_bytes;
                if (epos <= eb.size()) {
                    for (int ti = 0; ti < T; ti++) {
                        if (!((ep[bitmap_pos + static_cast<size_t>(ti) / 8]
                               >> (ti % 8)) & 1u))
                            continue;
                        long m = vbyte(ep, epos);
                        std::vector<int> positions;
                        if (sparse_rice) {
                            positions.resize(static_cast<size_t>(m));
                            int k = ef_l(m, t_df[ti]);
                            PackedBitReader reader(ep, epos);
                            int64_t previous = -1;
                            for (long i = 0; i < m; i++) {
                                uint32_t quotient = 0;
                                while (reader.read(1) != 0) quotient++;
                                uint32_t remainder = reader.read(k);
                                previous += static_cast<int64_t>(
                                    ((quotient << k) | remainder) + 1u);
                                positions[static_cast<size_t>(i)] =
                                    static_cast<int>(previous);
                            }
                            epos = reader.pos;
                        } else {
                            get_monotone_ef(
                                ep, epos, m, t_df[ti], positions);
                        }
                        auto &docs = t_early_ords[ti];
                        docs.reserve(positions.size());
                        for (int pi : positions)
                            if (pi >= 0 &&
                                static_cast<size_t>(pi) < t_ords[ti].size())
                                docs.push_back(t_ords[ti][pi]);
                    }
                }
            } else {
                size_t epos = 0;
                for (int ti = 0; ti < T; ti++) {
                    size_t df = static_cast<size_t>(t_df[ti]);
                    size_t bytes = (df + 7) / 8;
                    if (epos + bytes <= eb.size()) {
                        auto &docs = t_early_ords[ti];
                        for (size_t i = 0; i < df; i++)
                            if ((ep[epos + i / 8] >> (i % 8)) & 1u)
                                docs.push_back(t_ords[ti][i]);
                    }
                    epos += bytes;
                }
            }
        }
    }

    void build_document_norms() {
        int T = static_cast<int>(t_df.size());
        // VSM document norms: sqrt(sum_t (tf * log(N/df))^2).
        doc_norm.assign(N, 0.0);
        for (int ti = 0; ti < T; ti++) {
            double idf = std::log(static_cast<double>(N) / t_df[ti]);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                double w = tfs[i] * idf;
                doc_norm[ords[i]] += w * w;
            }
        }
        for (int d = 0; d < N; d++) doc_norm[d] = std::sqrt(doc_norm[d]);
    }

    void build_forward_index() {
        int T = static_cast<int>(t_df.size());
        // Forward index (doc -> [(term_ord, tf)]) for RM3, reconstructed in
        // memory as the TRANSPOSE of the inverted postings above rather than
        // read from disk. forward.bin held exactly the same (doc, term, tf)
        // triples as postings.bin, just grouped by doc instead of term — a
        // byte-for-byte redundant copy that was ~44% of the old index size, so
        // it is no longer persisted. We iterate terms in ascending term
        // ordinal, so each doc's term list is emitted ascending, identical to
        // what the old forward.bin decode produced -> RM3 output is unchanged.
        // Cost is two O(nnz) passes over already-decoded postings (~0.1s here);
        // this happens in load(), which is not part of the scored efficiency
        // metric (build time + query latency), so query latency is untouched.
        fwd_off.assign(N + 1, 0);
        // Pass 1: count each doc's terms into fwd_off[d+1].
        for (int ti = 0; ti < T; ti++)
            for (int o : t_ords[ti]) fwd_off[o + 1]++;
        // Prefix-sum to start offsets; fwd_off[N] is the total nnz.
        for (int d = 0; d < N; d++) fwd_off[d + 1] += fwd_off[d];
        size_t nnz = fwd_off[N];
        fwd_terms.resize(nnz);
        fwd_tfs.resize(nnz);
        // Pass 2: scatter. cursor[d] walks doc d's slice as we fill it.
        std::vector<size_t> cursor(fwd_off.begin(), fwd_off.end() - 1);
        for (int ti = 0; ti < T; ti++) {
            const auto &ords = t_ords[ti];
            const auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                size_t pos = cursor[ords[i]]++;
                fwd_terms[pos] = ti;
                fwd_tfs[pos] = tfs[i];
            }
        }
        has_forward = (nnz > 0);
    }

    void load(const std::string &dir) {
        load_documents(dir);
        load_postings(dir);
        load_early_matches(dir);
        build_document_norms();
        build_forward_index();
    }

    std::vector<OrdHit>
    topk_ord(std::vector<double> &score, std::vector<int> &touched, int k) {
        std::vector<std::pair<double, int>> items;
        items.reserve(touched.size());
        for (int d : touched) items.push_back({score[d], d});
        size_t kk = std::min(static_cast<size_t>(k), items.size());
        std::partial_sort(
            items.begin(), items.begin() + kk, items.end(),
            [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
        std::vector<OrdHit> out;
        out.reserve(kk);
        for (size_t i = 0; i < kk; i++)
            out.push_back({items[i].second, items[i].first});
        return out;
    }

    std::vector<DocHit> materialize(const std::vector<OrdHit> &hits) const {
        std::vector<DocHit> out;
        out.reserve(hits.size());
        for (const auto &hit : hits)
            out.push_back({docid[hit.first], hit.second});
        return out;
    }

    std::vector<OrdHit>
    bm25_ord(
        const std::vector<std::string> &tokens, double k1, double b, int k) {
        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it == term_id.end()) continue;
            int ti = it->second, df = t_df[ti];
            double idf = std::log((static_cast<double>(N) - df + 0.5) / (df + 0.5) + 1.0);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                int d = ords[i];
                double tf = tfs[i];
                if (score[d] == 0.0) touched.push_back(d);
                score[d] += idf * (tf * (k1 + 1)) /
                            (tf + k1 * (1 - b + b * doc_len[d] / avg_doc_len));
            }
        }
        return topk_ord(score, touched, k);
    }

    std::vector<DocHit>
    bm25(const std::vector<std::string> &tokens, double k1, double b, int k) {
        return materialize(bm25_ord(tokens, k1, b, k));
    }

    std::vector<OrdHit>
    vsm_ord(const std::vector<std::string> &tokens, int k) {
        std::unordered_map<std::string, int> qcount;
        for (const auto &t : tokens) qcount[t]++;

        std::vector<std::pair<int, double>> qv;  // (termid, weight)
        double qnorm = 0.0;
        for (const auto &kv : qcount) {
            auto it = term_id.find(kv.first);
            if (it == term_id.end()) continue;
            int ti = it->second;
            double idf = std::log(static_cast<double>(N) / t_df[ti]);
            double w = kv.second * idf;
            qv.push_back({ti, w});
            qnorm += w * w;
        }
        qnorm = std::sqrt(qnorm);
        std::vector<OrdHit> empty;
        if (qnorm == 0.0) return empty;

        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (const auto &kv : qv) {
            int ti = kv.first;
            double st = kv.second / qnorm;
            double idf = std::log(static_cast<double>(N) / t_df[ti]);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                int d = ords[i];
                double dw = tfs[i] * idf;
                if (score[d] == 0.0) touched.push_back(d);
                score[d] += st * (dw / doc_norm[d]);
            }
        }
        return topk_ord(score, touched, k);
    }

    std::vector<DocHit>
    vsm(const std::vector<std::string> &tokens, int k) {
        return materialize(vsm_ord(tokens, k));
    }

    // Coordination/coverage arm: rank documents by how many distinct query
    // terms they contain. The optional idf weighting rewards coverage of rare
    // terms while keeping the signal independent of corpus-specific labels.
    std::vector<OrdHit>
    coverage_ord(
        const std::vector<std::string> &tokens, int k, bool idf_weighted,
        double idf_power = 1.0) {
        std::unordered_set<int> seen;
        std::vector<int> qterms;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it == term_id.end()) continue;
            if (seen.insert(it->second).second) qterms.push_back(it->second);
        }
        std::vector<OrdHit> empty;
        if (qterms.empty()) return empty;

        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (int ti : qterms) {
            double w = idf_weighted
                ? std::pow(std::log(static_cast<double>(N) / t_df[ti]), idf_power)
                : 1.0;
            for (int d : t_ords[ti]) {
                if (score[d] == 0.0) touched.push_back(d);
                score[d] += w;
            }
        }
        return topk_ord(score, touched, k);
    }

    std::vector<DocHit>
    coverage(const std::vector<std::string> &tokens, int k, bool idf_weighted,
             double idf_power = 1.0) {
        return materialize(
            coverage_ord(tokens, k, idf_weighted, idf_power));
    }

    // Rank documents by the number of distinct query terms present in the
    // first 12 raw words. early.bin stores a sparse posting list for each
    // term, containing only the documents where that term occurs in the
    // prefix.
    std::vector<OrdHit>
    early_match_ord(const std::vector<std::string> &tokens, int k) {
        std::unordered_set<int> seen;
        std::vector<int> qterms;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it != term_id.end() && seen.insert(it->second).second)
                qterms.push_back(it->second);
        }
        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (int ti : qterms) {
            for (int d : t_early_ords[ti]) {
                if (score[d] == 0.0) touched.push_back(d);
                score[d] += 1.0;
            }
        }
        return topk_ord(score, touched, k);
    }

    std::vector<DocHit>
    early_match(const std::vector<std::string> &tokens, int k) {
        return materialize(early_match_ord(tokens, k));
    }

    // BM25 scoring of a *weighted* bag of query terms into `score`/`touched`.
    // Each (term id, weight) contributes weight * bm25_term(term, doc).
    void bm25_weighted(const std::vector<std::pair<int, double>> &qterms,
                       double k1, double b,
                       std::vector<double> &score, std::vector<int> &touched) {
        for (const auto &kv : qterms) {
            int ti = kv.first;
            double wq = kv.second;
            int df = t_df[ti];
            double idf = std::log((static_cast<double>(N) - df + 0.5) / (df + 0.5) + 1.0);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                int d = ords[i];
                double tf = tfs[i];
                if (score[d] == 0.0) touched.push_back(d);
                score[d] += wq * idf * (tf * (k1 + 1)) /
                            (tf + k1 * (1 - b + b * doc_len[d] / avg_doc_len));
            }
        }
    }

    // RM3 pseudo-relevance feedback. Round 1: BM25 top-R feedback docs. Build a
    // relevance model P(w|R) = sum_d P(d|q) * tf(w,d)/|d| over those docs (using
    // the forward index), keep the top-M terms. Interpolate
    // with the original query model: P(w|q') = lambda*P(w|q0) + (1-lambda)*P(w|R).
    // Round 2: weighted BM25 with that expanded query. Returns top-`cand`
    // (doc_id, score) for fusion, exactly like bm25().
    std::vector<OrdHit>
    rm3_ord(
        const std::vector<std::string> &tokens, int R, int M, double lambda_,
        double k1, double b, int cand, double fb_temp, bool novel_only,
        double k1_round2 = -1.0, double b_round2 = -1.0) {
        std::vector<OrdHit> empty;
        if (!has_forward) return empty;

        // Round 1 BM25.
        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        std::vector<std::pair<int, double>> q0;  // in-vocab original query terms
        {
            std::unordered_map<int, int> qc;
            for (const auto &tok : tokens) {
                auto it = term_id.find(tok);
                if (it != term_id.end()) qc[it->second]++;
            }
            for (auto &kv : qc)
                q0.push_back({kv.first, static_cast<double>(kv.second)});
        }
        bm25_weighted(q0, k1, b, score, touched);

        // Top-R feedback docs.
        std::vector<std::pair<double, int>> items;
        items.reserve(touched.size());
        for (int d : touched) items.push_back({score[d], d});
        size_t RR = std::min(static_cast<size_t>(R), items.size());
        auto by_score = [](const std::pair<double, int> &a,
                           const std::pair<double, int> &b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        };
        std::partial_sort(items.begin(), items.begin() + RR, items.end(), by_score);
        // Feedback-doc weights P(d|q): softmax of the BM25 score normalised by
        // the top score, temperature fb_temp. Dividing by smax makes the
        // weights scale-free, so one corpus's larger BM25 magnitudes don't
        // sharpen the distribution relative to another's. Raw score/sum lets a
        // single anomalous top document dominate the relevance model.
        std::vector<double> fbw(RR, 0.0);
        double smax = RR > 0 ? score[items[0].second] : 1.0;
        if (smax <= 0.0) smax = 1.0;
        for (size_t i = 0; i < RR; i++)
            fbw[i] = std::exp((score[items[i].second] / smax - 1.0) / fb_temp);
        double ssum = 0.0;
        for (size_t i = 0; i < RR; i++) ssum += fbw[i];
        if (ssum <= 0.0) ssum = 1.0;

        // Relevance model.
        // novel_only spends the M budget only on terms the original query does
        // not already contain — a q0 term re-selected here would just double up
        // on weight it already gets from the lambda anchor, at the cost of an
        // expansion slot. (A df cap on candidates was also tried and rejected:
        // stopwords are already dropped at index time, so the high-df survivors
        // are topical and discriminative.)
        std::unordered_set<int> q0set;
        if (novel_only) for (auto &kv : q0) q0set.insert(kv.first);

        std::unordered_map<int, double> rel;
        for (size_t i = 0; i < RR; i++) {
            int d = items[i].second;
            double pd = fbw[i] / ssum;
            int dl = doc_len[d] > 0 ? doc_len[d] : 1;
            for (size_t j = fwd_off[d]; j < fwd_off[d + 1]; j++) {
                int tw = fwd_terms[j];
                if (novel_only && q0set.count(tw)) continue;
                rel[tw] += pd * static_cast<double>(fwd_tfs[j]) / dl;
            }
        }

        // Top-M expansion terms.
        std::vector<std::pair<double, int>> rl;
        rl.reserve(rel.size());
        for (auto &kv : rel) rl.push_back({kv.second, kv.first});
        size_t MM = std::min(static_cast<size_t>(M), rl.size());
        std::partial_sort(
            rl.begin(), rl.begin() + MM, rl.end(),
            [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
        double fsum = 0.0;
        for (size_t i = 0; i < MM; i++) fsum += rl[i].first;
        if (fsum <= 0.0) fsum = 1.0;

        // Interpolated query model P(w|q') = lambda*P(w|q0) + (1-lambda)*P(w|F).
        std::unordered_map<int, double> qmodel;
        for (size_t i = 0; i < MM; i++)
            qmodel[rl[i].second] += (1.0 - lambda_) * (rl[i].first / fsum);
        double q0sum = 0.0;
        for (auto &kv : q0) q0sum += kv.second;
        if (q0sum <= 0.0) q0sum = 1.0;
        for (auto &kv : q0) qmodel[kv.first] += lambda_ * (kv.second / q0sum);

        std::vector<std::pair<int, double>> qexp(qmodel.begin(), qmodel.end());

        // Round 2 weighted BM25.
        std::fill(score.begin(), score.end(), 0.0);
        touched.clear();
        double rk1 = k1_round2 > 0.0 ? k1_round2 : k1;
        double rb = b_round2 >= 0.0 ? b_round2 : b;
        bm25_weighted(qexp, rk1, rb, score, touched);
        return topk_ord(score, touched, cand);
    }

    std::vector<DocHit>
    rm3(const std::vector<std::string> &tokens, int R, int M, double lambda_,
        double k1, double b, int cand,
        double fb_temp, bool novel_only,
        double k1_round2 = -1.0, double b_round2 = -1.0) {
        return materialize(
            rm3_ord(tokens, R, M, lambda_, k1, b, cand, fb_temp,
                    novel_only, k1_round2, b_round2));
    }

    // Run the two fusion arms (RM3-or-BM25 and VSM) concurrently. Both arms
    // only read shared, immutable arrays (t_ords/t_tfs/doc_len/doc_norm/fwd_*)
    // and each owns its score/touched buffers, so concurrent reads are safe
    // with no locking. The VSM arm runs on a worker thread while this thread
    // runs the (heavier) RM3/BM25 arm. Called with the GIL released (see the
    // pybind def), so the two C++ threads run truly in parallel. Returns
    // {bm_hits, vs_hits}; retrieve() consumes these internally, while the
    // binding remains exposed for tuning scripts.
    std::pair<std::vector<OrdHit>, std::vector<OrdHit>>
    arms_ord(
        const std::vector<std::string> &tokens, bool use_rm3,
        int R, int M, double lambda_, double k1, double b, int cand,
        double fb_temp, bool novel_only,
        double k1_round2 = -1.0, double b_round2 = -1.0) {
        std::vector<OrdHit> vs_hits;
        std::thread vt([&] { vs_hits = vsm_ord(tokens, cand); });
        std::vector<OrdHit> bm_hits =
            (use_rm3 && has_forward)
                ? rm3_ord(
                    tokens, R, M, lambda_, k1, b, cand, fb_temp,
                    novel_only, k1_round2, b_round2)
                : bm25_ord(tokens, k1, b, cand);
        vt.join();
        return {bm_hits, vs_hits};
    }

    std::pair<std::vector<DocHit>, std::vector<DocHit>>
    arms(const std::vector<std::string> &tokens, bool use_rm3,
         int R, int M, double lambda_, double k1, double b, int cand,
         double fb_temp, bool novel_only,
         double k1_round2 = -1.0, double b_round2 = -1.0) {
        auto hits = arms_ord(
            tokens, use_rm3, R, M, lambda_, k1, b, cand, fb_temp,
            novel_only, k1_round2, b_round2);
        return {materialize(hits.first), materialize(hits.second)};
    }

    // Complete query pipeline behind one pybind boundary. The raw query enters
    // once and only the final top-k leaves: tokenisation, both retrieval arms,
    // coverage/early signals, RRF fusion, and reranking all stay in C++.
    std::vector<DocHit>
    retrieve(const std::string &query, int k, bool use_rm3,
             int R, int M, double lambda_, double k1, double b, int cand,
             double fb_temp, bool novel_only,
             double k1_round2, double b_round2,
             int rrf_k, double w_bm,
             double coverage_alpha, int rerank_depth,
             double early_alpha) {
        std::vector<std::string> raw_tokens;
        query_tok.tokenize(query, raw_tokens);

        // Match indexer.tokenize(): lowercase terms survive even when OOV;
        // case-sensitive terms survive only if pruning kept them in the index.
        std::vector<std::string> tokens;
        tokens.reserve(raw_tokens.size());
        for (const auto &tok : raw_tokens) {
            bool lowercase = true;
            for (unsigned char c : tok) {
                if (c >= 'A' && c <= 'Z') {
                    lowercase = false;
                    break;
                }
            }
            if (lowercase || term_id.find(tok) != term_id.end())
                tokens.push_back(tok);
        }

        // arms() already overlaps RM3/BM25 and VSM. Coverage plus early are
        // substantially cheaper than either main arm, so one feature worker
        // can finish both while the main arms run, avoiding another thread's
        // startup/contention cost. Every task only reads immutable index arrays
        // and owns its score/touched buffers, so no locking is needed and the
        // exact top-500 feature semantics are preserved.
        std::vector<OrdHit> coverage_hits;
        std::vector<OrdHit> early_hits;
        std::thread feature_thread([&] {
            coverage_hits = coverage_ord(tokens, cand, true);
            early_hits = early_match_ord(tokens, cand);
        });
        auto arm_hits = arms_ord(
            tokens, use_rm3, R, M, lambda_, k1, b, cand, fb_temp,
            novel_only, k1_round2, b_round2);
        feature_thread.join();

        std::unordered_map<int, double> fused_score;
        fused_score.reserve(
            arm_hits.first.size() + arm_hits.second.size());
        auto add_hits = [&](const std::vector<OrdHit> &hits, double weight) {
            for (size_t i = 0; i < hits.size(); i++)
                fused_score[hits[i].first] +=
                    weight / static_cast<double>(rrf_k + i + 1);
        };
        add_hits(arm_hits.first, w_bm);
        add_hits(arm_hits.second, 1.0 - w_bm);

        std::vector<OrdHit> ranked;
        ranked.reserve(fused_score.size());
        for (const auto &hit : fused_score)
            ranked.push_back(hit);
        auto rank_order = [&](const OrdHit &a, const OrdHit &b) {
            return a.second > b.second ||
                   (a.second == b.second &&
                    docid[a.first] < docid[b.first]);
        };
        std::sort(ranked.begin(), ranked.end(), rank_order);

        if (coverage_alpha > 0.0) {
            std::unordered_map<int, double> coverage_score;
            std::unordered_map<int, double> early_score;
            for (const auto &hit : coverage_hits)
                coverage_score[hit.first] = hit.second;
            for (const auto &hit : early_hits)
                early_score[hit.first] = hit.second;

            size_t depth = std::min(
                ranked.size(),
                static_cast<size_t>(std::max(0, rerank_depth)));
            double fmax = depth > 0 ? ranked[0].second : 0.0;
            double cmax = 0.0;
            for (size_t i = 0; i < depth; i++) {
                auto it = coverage_score.find(ranked[i].first);
                if (it != coverage_score.end())
                    cmax = std::max(cmax, it->second);
            }
            if (fmax > 0.0 && cmax > 0.0) {
                for (size_t i = 0; i < depth; i++) {
                    double cov = 0.0;
                    auto it = coverage_score.find(ranked[i].first);
                    if (it != coverage_score.end()) cov = it->second;
                    ranked[i].second =
                        ranked[i].second / fmax + coverage_alpha * cov / cmax;
                }
                std::sort(ranked.begin(), ranked.begin() + depth, rank_order);
            }

            if (early_alpha > 0.0) {
                std::unordered_set<std::string> unique_tokens(
                    tokens.begin(), tokens.end());
                double qden = static_cast<double>(
                    std::max<size_t>(1, unique_tokens.size()));
                depth = std::min(
                    ranked.size(),
                    static_cast<size_t>(std::max(0, rerank_depth)));
                for (size_t i = 0; i < depth; i++) {
                    double early = 0.0;
                    auto it = early_score.find(ranked[i].first);
                    if (it != early_score.end()) early = it->second;
                    ranked[i].second += early_alpha * early / qden;
                }
                std::sort(ranked.begin(), ranked.begin() + depth, rank_order);
            }
        }

        if (k < 0) k = 0;
        if (ranked.size() > static_cast<size_t>(k))
            ranked.resize(static_cast<size_t>(k));
        return materialize(ranked);
    }
};

PYBIND11_MODULE(_index_cpp, m) {
    m.doc() = "Native C++ inverted index: BM25 + VSM over the compressed postings.";
    py::class_<NativeIndex>(m, "NativeIndex")
        .def(py::init<const std::string &, const std::vector<std::string> &>(),
             py::arg("index_dir"),
             py::arg("stopword_stems") = std::vector<std::string>{})
        .def("bm25", &NativeIndex::bm25,
             py::arg("tokens"), py::arg("k1"), py::arg("b"), py::arg("k"))
        .def("coverage", &NativeIndex::coverage,
             py::arg("tokens"), py::arg("k"), py::arg("idf_weighted"),
             py::arg("idf_power") = 1.0)
        .def("early_match", &NativeIndex::early_match,
             py::arg("tokens"), py::arg("k"))
        .def("vsm", &NativeIndex::vsm, py::arg("tokens"), py::arg("k"))
        .def("rm3", &NativeIndex::rm3,
             py::arg("tokens"), py::arg("R"), py::arg("M"), py::arg("lambda_"),
             py::arg("k1"), py::arg("b"), py::arg("cand"),
             py::arg("fb_temp"), py::arg("novel_only"),
             py::arg("k1_round2") = -1.0,
             py::arg("b_round2") = -1.0)
        .def("arms", &NativeIndex::arms,
             py::arg("tokens"), py::arg("use_rm3"), py::arg("R"), py::arg("M"),
             py::arg("lambda_"), py::arg("k1"), py::arg("b"), py::arg("cand"),
             py::arg("fb_temp"), py::arg("novel_only"),
             py::arg("k1_round2") = -1.0,
             py::arg("b_round2") = -1.0,
             py::call_guard<py::gil_scoped_release>())
        .def("retrieve", &NativeIndex::retrieve,
             py::arg("query"), py::arg("k"), py::arg("use_rm3"),
             py::arg("R"), py::arg("M"), py::arg("lambda_"),
             py::arg("k1"), py::arg("b"), py::arg("cand"),
             py::arg("fb_temp"), py::arg("novel_only"),
             py::arg("k1_round2"), py::arg("b_round2"),
             py::arg("rrf_k"), py::arg("w_bm"),
             py::arg("coverage_alpha"), py::arg("rerank_depth"),
             py::arg("early_alpha"),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("has_forward",
                               [](const NativeIndex &n) { return n.has_forward; });
}
