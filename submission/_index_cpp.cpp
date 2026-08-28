// submission/_index_cpp.cpp — optional native (C++) index for the hot paths.
//
// Loads the compressed index (docs.txt / terms.txt / postings.bin written by
// InvertedIndex.save_v2) directly into contiguous C++ arrays, and scores BM25
// and VSM over them. This replaces BOTH slow Python steps at once:
//   (a) the per-query scoring loops (bm25.py / boolean_vsm.py), and
//   (b) the load-time reconstruction of ~17M Python dict entries in load_v2.
//
// Tokenisation stays in Python (Snowball stemmer): the caller passes the
// already-tokenised query terms. Fusion (RRF) also stays in Python — it is a
// cheap merge of two top-k lists. This module only owns the two hot loops.
//
// Drop-in + optional: retrieve.py uses it if built and falls back to the pure
// Python InvertedIndex + bm25/boolean_vsm path otherwise. Formulas here match
// bm25.py and boolean_vsm.py exactly so rankings are identical (up to
// tie-breaking).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace py = pybind11;

struct NativeIndex {
    int N = 0;
    double avg_doc_len = 0.0;
    std::vector<std::string> docid;   // ordinal -> doc_id string
    std::vector<int> doc_len;         // ordinal -> length
    std::vector<double> doc_norm;     // ordinal -> sqrt(VSM norm)
    std::unordered_map<std::string, int> term_id;
    std::vector<std::vector<int>> t_ords;  // term -> sorted doc ordinals
    std::vector<std::vector<int>> t_tfs;   // term -> term frequencies
    std::vector<int> t_df;                 // term -> document frequency

    // Forward index (CSR): for RM3, doc d's terms are fwd_terms[fwd_off[d] ..
    // fwd_off[d+1]) with parallel fwd_tfs. Absent unless forward.bin exists.
    bool has_forward = false;
    std::vector<size_t> fwd_off;
    std::vector<int> fwd_terms;
    std::vector<int> fwd_tfs;

    explicit NativeIndex(const std::string &dir) { load(dir); }

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

    // Inverse of _spimi_cpp.cpp's put_postings (A1 codec): `df` codes
    // (gap*2 + tf>1 flag), then only the tfs for flagged postings.
    static void get_postings(const unsigned char *p, size_t &pos, long df,
                              std::vector<int> &ords, std::vector<int> &tfs) {
        ords.resize(df);
        tfs.resize(df);
        std::vector<unsigned char> flags(df);
        long prev = 0;
        for (long i = 0; i < df; i++) {
            long code = vbyte(p, pos);
            long gap = code >> 1;
            flags[i] = static_cast<unsigned char>(code & 1);
            prev = (i == 0) ? gap : prev + gap;
            ords[i] = static_cast<int>(prev);
        }
        for (long i = 0; i < df; i++)
            tfs[i] = flags[i] ? static_cast<int>(vbyte(p, pos)) : 1;
    }

    void load(const std::string &dir) {
        // A4: docs.txt now holds ONLY verbatim doc-id lines; lengths live in
        // the parallel VByte stream doclen.bin, same doc order.
        {
            std::ifstream f(dir + "/docs.txt");
            std::string line;
            while (std::getline(f, line)) docid.push_back(line);
        }
        N = static_cast<int>(docid.size());
        {
            std::ifstream f(dir + "/doclen.bin", std::ios::binary);
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string db = ss.str();
            const unsigned char *dp = reinterpret_cast<const unsigned char *>(db.data());
            size_t dpos = 0;
            doc_len.reserve(N);
            for (int d = 0; d < N; d++) doc_len.push_back(static_cast<int>(vbyte(dp, dpos)));
        }
        long tot = 0;
        for (int L : doc_len) tot += L;
        avg_doc_len = static_cast<double>(tot) / N;

        std::string blob;
        {
            std::ifstream f(dir + "/postings.bin", std::ios::binary);
            std::ostringstream ss;
            ss << f.rdbuf();
            blob = ss.str();
        }
        const unsigned char *p = reinterpret_cast<const unsigned char *>(blob.data());
        size_t pos = 0;

        std::vector<std::string> terms;
        std::vector<int> dfs;
        {
            std::ifstream f(dir + "/terms.txt");
            std::string line;
            while (std::getline(f, line)) {
                size_t tab = line.find('\t');
                terms.push_back(line.substr(0, tab));
                dfs.push_back(std::stoi(line.substr(tab + 1)));
            }
        }
        int T = static_cast<int>(terms.size());
        t_ords.resize(T);
        t_tfs.resize(T);
        t_df.resize(T);
        for (int ti = 0; ti < T; ti++) {
            int df = dfs[ti];
            t_df[ti] = df;
            term_id[terms[ti]] = ti;
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            get_postings(p, pos, df, ords, tfs);  // A1 codec
        }

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

        // Forward index (optional): per doc, VByte(count), term-ord gaps, tfs.
        {
            std::ifstream f(dir + "/forward.bin", std::ios::binary);
            if (f) {
                std::ostringstream ss;
                ss << f.rdbuf();
                std::string fb = ss.str();
                if (!fb.empty()) {
                    const unsigned char *fp = reinterpret_cast<const unsigned char *>(fb.data());
                    size_t fpos = 0;
                    fwd_off.assign(N + 1, 0);
                    for (int d = 0; d < N; d++) {
                        long n = vbyte(fp, fpos);
                        std::vector<int> ords, tfs;
                        get_postings(fp, fpos, n, ords, tfs);  // A1 codec
                        fwd_terms.insert(fwd_terms.end(), ords.begin(), ords.end());
                        fwd_tfs.insert(fwd_tfs.end(), tfs.begin(), tfs.end());
                        fwd_off[d + 1] = fwd_terms.size();
                    }
                    has_forward = true;
                }
            }
        }
    }

    std::vector<std::pair<std::string, double>>
    topk(std::vector<double> &score, std::vector<int> &touched, int k) {
        std::vector<std::pair<double, int>> items;
        items.reserve(touched.size());
        for (int d : touched) items.push_back({score[d], d});
        size_t kk = std::min(static_cast<size_t>(k), items.size());
        std::partial_sort(
            items.begin(), items.begin() + kk, items.end(),
            [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
        std::vector<std::pair<std::string, double>> out;
        out.reserve(kk);
        for (size_t i = 0; i < kk; i++)
            out.push_back({docid[items[i].second], items[i].first});
        return out;
    }

    std::vector<std::pair<std::string, double>>
    bm25(const std::vector<std::string> &tokens, double k1, double b, int k) {
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
        return topk(score, touched, k);
    }

    std::vector<std::pair<std::string, double>>
    vsm(const std::vector<std::string> &tokens, int k) {
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
        std::vector<std::pair<std::string, double>> empty;
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
        return topk(score, touched, k);
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
    std::vector<std::pair<std::string, double>>
    rm3(const std::vector<std::string> &tokens, int R, int M, double lambda_,
        double k1, double b, int cand) {
        std::vector<std::pair<std::string, double>> empty;
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
            for (auto &kv : qc) q0.push_back({kv.first, static_cast<double>(kv.second)});
        }
        bm25_weighted(q0, k1, b, score, touched);

        // Top-R feedback docs.
        std::vector<std::pair<double, int>> items;
        items.reserve(touched.size());
        for (int d : touched) items.push_back({score[d], d});
        size_t RR = std::min(static_cast<size_t>(R), items.size());
        std::partial_sort(
            items.begin(), items.begin() + RR, items.end(),
            [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
        double ssum = 0.0;
        for (size_t i = 0; i < RR; i++) ssum += items[i].first;
        if (ssum <= 0.0) ssum = 1.0;

        // Relevance model.
        std::unordered_map<int, double> rel;
        for (size_t i = 0; i < RR; i++) {
            int d = items[i].second;
            double pd = items[i].first / ssum;
            int dl = doc_len[d] > 0 ? doc_len[d] : 1;
            for (size_t j = fwd_off[d]; j < fwd_off[d + 1]; j++) {
                int tw = fwd_terms[j];
                rel[tw] += pd * (static_cast<double>(fwd_tfs[j]) / dl);
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
        bm25_weighted(qexp, k1, b, score, touched);
        return topk(score, touched, cand);
    }
};

PYBIND11_MODULE(_index_cpp, m) {
    m.doc() = "Native C++ inverted index: BM25 + VSM over the compressed postings.";
    py::class_<NativeIndex>(m, "NativeIndex")
        .def(py::init<const std::string &>())
        .def("bm25", &NativeIndex::bm25,
             py::arg("tokens"), py::arg("k1"), py::arg("b"), py::arg("k"))
        .def("vsm", &NativeIndex::vsm, py::arg("tokens"), py::arg("k"))
        .def("rm3", &NativeIndex::rm3,
             py::arg("tokens"), py::arg("R"), py::arg("M"), py::arg("lambda_"),
             py::arg("k1"), py::arg("b"), py::arg("cand"))
        .def_property_readonly("has_forward",
                               [](const NativeIndex &n) { return n.has_forward; });
}
