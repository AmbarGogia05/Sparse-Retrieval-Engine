// submission/_spimi_cpp.cpp — SPIMI index builder in C++ (optional).
//
// Single-pass in-memory indexing with block flushing: accumulate postings in
// memory and, when the posting count crosses a threshold, flush a sorted
// block to a temp file. finalize() k-way merges the blocks, applies the
// truecase df-ratio gate, and writes the SAME compressed on-disk format as
// InvertedIndex.save_v2 (docs.txt / terms.txt / postings.bin / meta.json) so
// NativeIndex / load_v2 read it unchanged.
//
// Purpose is OOM-safety on large corpora: peak memory is one block plus the
// streaming merge, never the whole postings set. Tokenisation stays in Python
// (memoised nltk Snowball) — the caller passes each doc's tokens.
//
// Optional + drop-in: retrieve.build_index() uses this if built and falls
// back to InvertedIndex().build()+save_v2() otherwise.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>

namespace py = pybind11;

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

template <typename T>
static void wr(std::ofstream &f, T v) { f.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
template <typename T>
static T rd(std::ifstream &f) { T v; f.read(reinterpret_cast<char *>(&v), sizeof(T)); return v; }

// A block on disk: records sorted by term, each
//   [u32 term_len][term bytes][u32 df][df x u32 ord][df x u32 tf]
struct BlockReader {
    std::ifstream f;
    bool ok = false;
    std::string term;
    std::vector<uint32_t> ords, tfs;

    explicit BlockReader(const std::string &path) : f(path, std::ios::binary) { advance(); }

    void advance() {
        uint32_t tl;
        if (!f.read(reinterpret_cast<char *>(&tl), sizeof(tl))) { ok = false; return; }
        term.resize(tl);
        f.read(&term[0], tl);
        uint32_t df = rd<uint32_t>(f);
        ords.resize(df);
        tfs.resize(df);
        f.read(reinterpret_cast<char *>(ords.data()), df * sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(tfs.data()), df * sizeof(uint32_t));
        ok = true;
    }
};

struct SpimiBuilder {
    std::string dir;
    size_t flush_threshold;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> block;
    size_t cur_postings = 0;
    uint32_t n_docs = 0;
    long total_len = 0;
    std::vector<std::string> block_files;
    std::ofstream docs_out;
    std::unordered_set<std::string> case_terms_seen;

    SpimiBuilder(const std::string &d, size_t threshold)
        : dir(d), flush_threshold(threshold), docs_out(d + "/docs.txt") {}

    static bool has_upper(const std::string &s) {
        for (char c : s)
            if (c >= 'A' && c <= 'Z') return true;
        return false;
    }

    void add_document(const std::string &doc_id, const std::vector<std::string> &tokens, int doc_len) {
        docs_out << doc_id << '\t' << doc_len << '\n';
        uint32_t ord = n_docs++;
        total_len += doc_len;

        std::unordered_map<std::string, uint32_t> tf;
        for (const auto &t : tokens) {
            tf[t]++;
            if (has_upper(t)) case_terms_seen.insert(t);
        }
        for (auto &kv : tf) {
            auto &vec = block[kv.first];
            vec.push_back({ord, kv.second});
            cur_postings++;
        }
        if (cur_postings >= flush_threshold) flush_block();
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
        }
        block.clear();
        cur_postings = 0;
        block_files.push_back(path);
    }

    std::vector<std::string> case_terms() {
        return std::vector<std::string>(case_terms_seen.begin(), case_terms_seen.end());
    }

    void finalize(const std::unordered_map<std::string, std::string> &canonical, double df_ratio) {
        flush_block();
        docs_out.close();

        // Pass 1: global df per term (header scan, skipping postings bytes).
        std::unordered_map<std::string, uint32_t> df_all;
        for (auto &path : block_files) {
            std::ifstream f(path, std::ios::binary);
            uint32_t tl;
            while (f.read(reinterpret_cast<char *>(&tl), sizeof(tl))) {
                std::string term(tl, '\0');
                f.read(&term[0], tl);
                uint32_t df = rd<uint32_t>(f);
                df_all[term] += df;
                f.seekg(static_cast<std::streamoff>(df) * 2 * sizeof(uint32_t), std::ios::cur);
            }
        }

        // Truecase gate: keep a case term only if df(case)/df(canonical) <= ratio.
        std::unordered_set<std::string> prune, keep_case;
        for (auto &kv : df_all) {
            const std::string &term = kv.first;
            if (!has_upper(term)) continue;
            auto ci = canonical.find(term);
            uint32_t cdf = 0;
            if (ci != canonical.end()) {
                auto di = df_all.find(ci->second);
                if (di != df_all.end()) cdf = di->second;
            }
            if (cdf == 0 || static_cast<double>(kv.second) / cdf <= df_ratio)
                keep_case.insert(term);
            else
                prune.insert(term);
        }

        // Pass 2: k-way merge by term, write terms.txt + postings.bin.
        std::vector<BlockReader *> readers;
        for (auto &path : block_files) readers.push_back(new BlockReader(path));

        std::ofstream terms_out(dir + "/terms.txt");
        std::string postings;  // buffered, written once
        bool first_term = true;

        while (true) {
            // smallest current term across readers
            std::string mterm;
            bool found = false;
            for (auto *r : readers)
                if (r->ok && (!found || r->term < mterm)) { mterm = r->term; found = true; }
            if (!found) break;

            // gather postings for mterm from all readers (block order => ascending ords)
            std::vector<uint32_t> ords, tfs;
            for (auto *r : readers) {
                if (r->ok && r->term == mterm) {
                    ords.insert(ords.end(), r->ords.begin(), r->ords.end());
                    tfs.insert(tfs.end(), r->tfs.begin(), r->tfs.end());
                    r->advance();
                }
            }
            if (prune.count(mterm)) continue;

            if (!first_term) terms_out << '\n';
            first_term = false;
            terms_out << mterm << '\t' << ords.size();

            uint32_t prev = 0;
            for (size_t i = 0; i < ords.size(); i++) {
                put_vbyte(postings, i == 0 ? ords[i] : ords[i] - prev);
                prev = ords[i];
            }
            for (uint32_t tf : tfs) put_vbyte(postings, tf);
        }

        for (auto *r : readers) delete r;
        std::ofstream(dir + "/postings.bin", std::ios::binary).write(postings.data(), postings.size());

        // meta.json (N, avg_doc_len, case_terms) — hand-written, no JSON dep.
        double avg = n_docs ? static_cast<double>(total_len) / n_docs : 0.0;
        std::ofstream meta(dir + "/meta.json");
        meta << std::setprecision(std::numeric_limits<double>::max_digits10);
        meta << "{\"N\": " << n_docs << ", \"avg_doc_len\": " << avg << ", \"case_terms\": [";
        bool f2 = true;
        for (auto &t : keep_case) {
            if (!f2) meta << ", ";
            f2 = false;
            meta << '"' << t << '"';
        }
        meta << "]}";
        meta.close();

        for (auto &path : block_files) std::remove(path.c_str());
    }
};

PYBIND11_MODULE(_spimi_cpp, m) {
    m.doc() = "SPIMI index builder: block-flushing, writes save_v2's format.";
    py::class_<SpimiBuilder>(m, "SpimiBuilder")
        .def(py::init<const std::string &, size_t>(), py::arg("index_dir"), py::arg("flush_threshold"))
        .def("add_document", &SpimiBuilder::add_document,
             py::arg("doc_id"), py::arg("tokens"), py::arg("doc_len"))
        .def("case_terms", &SpimiBuilder::case_terms)
        .def("finalize", &SpimiBuilder::finalize, py::arg("canonical"), py::arg("df_ratio"));
}
