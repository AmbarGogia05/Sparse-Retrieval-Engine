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
#include <cstdint>
#include <thread>

namespace py = pybind11;

struct NativeIndex {
    int N = 0;
    double avg_doc_len = 0.0;
    std::vector<std::string> docid;   // ordinal -> doc_id string
    std::unordered_map<std::string, int> doc_ord;
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

    // floor(log2(N/m)) == bit_length(N/m)-1; the Elias-Fano low-bit width.
    static int ef_l(long m, long N) {
        if (m <= 0) return 0;
        long q = N / m;
        if (q < 1) return 0;
        int l = 0;
        while ((q >> 1) != 0) { q >>= 1; l++; }
        return l;
    }

    // Inverse of _spimi_cpp.cpp's put_postings_ef / indexer._encode_postings
    // (Elias-Fano of the monotone doc-ids, then (tf-1) in 128-blocks). See the
    // codec comment in indexer.py; l is recomputed from (df, N), not stored.
    static void get_postings(const unsigned char *p, size_t &pos, long df, long N,
                             std::vector<int> &ords, std::vector<int> &tfs,
                             bool all_tf_one = false) {
        long m = df;
        ords.resize(m);
        tfs.resize(m);
        int l = ef_l(m, N);
        // Low bits: m values of l bits each, LSB-first, byte-padded.
        std::vector<int> lows(m, 0);
        if (l > 0) {
            uint64_t buf = 0;
            int nb = 0;
            size_t bi = pos;
            long lmask = (1L << l) - 1;
            for (long i = 0; i < m; i++) {
                while (nb < l) { buf |= static_cast<uint64_t>(p[bi++]) << nb; nb += 8; }
                lows[i] = static_cast<int>(buf & lmask);
                buf >>= l;
                nb -= l;
            }
        }
        pos += static_cast<size_t>((m * static_cast<long>(l) + 7) / 8);
        // High bits: read bits until m one-bits; each 0 raises the high part.
        {
            uint64_t buf = 0;
            int nb = 0;
            size_t bi = pos;
            long cur = 0;
            long ones = 0;
            while (ones < m) {
                if (nb == 0) { buf = p[bi++]; nb = 8; }
                int bit = static_cast<int>(buf & 1);
                buf >>= 1;
                nb--;
                if (bit) {
                    ords[ones] = static_cast<int>((cur << l) | static_cast<long>(lows[ones]));
                    ones++;
                } else {
                    cur++;
                }
            }
            pos = bi;  // writer byte-padded the high stream
        }
        if (all_tf_one) {
            tfs.assign(m, 1);
            return;
        }
        // tf blocks: (tf-1) in blocks of 128, per block a width byte then packed.
        long done = 0;
        while (done < m) {
            long k = std::min(static_cast<long>(128), m - done);
            int width = p[pos++];
            if (width == 0) {
                for (long j = 0; j < k; j++) tfs[done + j] = 1;
            } else {
                uint64_t buf = 0;
                int nb = 0;
                long wmask = (1L << width) - 1;
                for (long j = 0; j < k; j++) {
                    while (nb < width) { buf |= static_cast<uint64_t>(p[pos++]) << nb; nb += 8; }
                    tfs[done + j] = static_cast<int>((buf & wmask) + 1);
                    buf >>= width;
                    nb -= width;
                }
            }
            done += k;
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

    void load(const std::string &dir) {
        // A4: docs.txt now holds ONLY verbatim doc-id lines; lengths live in
        // the parallel VByte stream doclen.bin, same doc order.
        {
            std::ifstream packed(dir + "/docids.bin", std::ios::binary);
            char magic[4] = {};
            if (packed.read(magic, 4) &&
                std::string(magic, 4) == std::string("D36\1", 4)) {
                uint64_t count = 0;
                packed.seekg(0, std::ios::end);
                std::streamoff total = packed.tellg();
                packed.seekg(4, std::ios::beg);
                count = total >= 4 ? static_cast<uint64_t>((total - 4) / 6) : 0;
                docid.reserve(static_cast<size_t>(count));
                for (uint64_t i = 0; i < count; i++) {
                    uint64_t value = 0;
                    for (int j = 0; j < 6; j++)
                        value |= static_cast<uint64_t>(
                            static_cast<unsigned char>(packed.get())) << (8 * j);
                    std::string id(8, '0');
                    for (int j = 7; j >= 0; j--) {
                        int digit = static_cast<int>(value % 36u);
                        id[j] = digit < 10 ? static_cast<char>('0' + digit)
                                           : static_cast<char>('a' + digit - 10);
                        value /= 36u;
                    }
                    docid.push_back(std::move(id));
                }
            } else {
                std::ifstream f(dir + "/docs.txt");
                std::string line;
                while (std::getline(f, line)) docid.push_back(line);
            }
        }
        N = static_cast<int>(docid.size());
        for (int d = 0; d < N; d++) doc_ord[docid[d]] = d;
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

        // terms.txt is front-coded (alphabetical): per term VByte(shared prefix
        // len) VByte(suffix len) suffix VByte(df), reconstructed against the
        // previous term. Mirrors indexer._decode_terms / _spimi put_term_frontcoded.
        std::vector<std::string> terms;
        std::vector<int> dfs;
        {
            std::ifstream f(dir + "/terms.txt", std::ios::binary);
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string tb = ss.str();
            const unsigned char *tp = reinterpret_cast<const unsigned char *>(tb.data());
            size_t tpos = 0, tn = tb.size();
            std::string prev;
            while (tpos < tn) {
                long shared = vbyte(tp, tpos);
                long suflen = vbyte(tp, tpos);
                std::string term = prev.substr(0, static_cast<size_t>(shared));
                term.append(reinterpret_cast<const char *>(tp + tpos), static_cast<size_t>(suflen));
                tpos += static_cast<size_t>(suflen);
                long df = vbyte(tp, tpos);
                terms.push_back(term);
                dfs.push_back(static_cast<int>(df));
                prev = term;
            }
        }
        int T = static_cast<int>(terms.size());
        t_ords.resize(T);
        t_tfs.resize(T);
        t_early_ords.resize(T);
        t_df.resize(T);
        std::vector<uint8_t> tf_all_one(T, 0);
        {
            std::ifstream f(dir + "/tf1.bin", std::ios::binary);
            std::string tfb((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
            size_t bytes = (static_cast<size_t>(T) + 7) / 8;
            if (tfb.size() >= 3 + bytes && tfb.compare(0, 3, "TF1") == 0) {
                for (int ti = 0; ti < T; ti++)
                    tf_all_one[ti] = static_cast<uint8_t>(
                        (static_cast<unsigned char>(tfb[3 + ti / 8])
                         >> (ti % 8)) & 1u);
            }
        }
        for (int ti = 0; ti < T; ti++) {
            int df = dfs[ti];
            t_df[ti] = df;
            term_id[terms[ti]] = ti;
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            get_postings(p, pos, df, N, ords, tfs,
                         tf_all_one[ti] != 0);  // Elias-Fano codec
        }

        // Optional prefix-presence stream. New indexes store sparse posting
        // positions as Elias-Fano records ("SEF1"); old packed-bit indexes are
        // still accepted. In RAM retain only the matching document ordinals,
        // so early_match traverses the sparse signal rather than full postings.
        {
            std::ifstream f(dir + "/early.bin", std::ios::binary);
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string eb = ss.str();
            const unsigned char *ep =
                reinterpret_cast<const unsigned char *>(eb.data());
            if (eb.size() >= 4 && eb.compare(0, 4, "SEF1") == 0) {
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
                        get_monotone_ef(ep, epos, m, t_df[ti], positions);
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
        {
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
    bm25plus(const std::vector<std::string> &tokens, double k1, double b,
             int k, double delta) {
        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it == term_id.end()) continue;
            int ti = it->second, df = t_df[ti];
            double idf = std::log((static_cast<double>(N) - df + 0.5) /
                                  (df + 0.5) + 1.0);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                int d = ords[i];
                double tf = tfs[i];
                if (score[d] == 0.0) touched.push_back(d);
                double norm = 1.0 - b + b * doc_len[d] / avg_doc_len;
                score[d] += idf * ((tf * (k1 + 1.0)) /
                                   (tf + k1 * norm) + delta);
            }
        }
        return topk(score, touched, k);
    }

    std::vector<std::pair<std::string, double>>
    bm25l(const std::vector<std::string> &tokens, double k1, double b, int k) {
        std::vector<double> score(N, 0.0);
        std::vector<int> touched;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it == term_id.end()) continue;
            int ti = it->second, df = t_df[ti];
            double idf = std::log((static_cast<double>(N) - df + 0.5) /
                                  (df + 0.5) + 1.0);
            auto &ords = t_ords[ti];
            auto &tfs = t_tfs[ti];
            for (size_t i = 0; i < ords.size(); i++) {
                int d = ords[i];
                double tf = tfs[i];
                if (score[d] == 0.0) touched.push_back(d);
                double norm = 1.0 - b + b * doc_len[d] / avg_doc_len;
                double tf_norm = tf / norm;
                score[d] += idf * ((k1 + 1.0) * tf_norm /
                                   (k1 + tf_norm));
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

    // Coordination/coverage arm: rank documents by how many distinct query
    // terms they contain. The optional idf weighting rewards coverage of rare
    // terms while keeping the signal independent of corpus-specific labels.
    std::vector<std::pair<std::string, double>>
    coverage(const std::vector<std::string> &tokens, int k, bool idf_weighted,
             double idf_power = 1.0) {
        std::unordered_set<int> seen;
        std::vector<int> qterms;
        for (const auto &tok : tokens) {
            auto it = term_id.find(tok);
            if (it == term_id.end()) continue;
            if (seen.insert(it->second).second) qterms.push_back(it->second);
        }
        std::vector<std::pair<std::string, double>> empty;
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
        return topk(score, touched, k);
    }

    // Rank documents by the number of distinct query terms present in the
    // first 12 raw words. Each ordinary posting has one aligned packed flag.
    std::vector<std::pair<std::string, double>>
    early_match(const std::vector<std::string> &tokens, int k) {
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
        return topk(score, touched, k);
    }

    int document_length(const std::string &id) const {
        auto it = doc_ord.find(id);
        return it == doc_ord.end() ? 0 : doc_len[it->second];
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
        double k1, double b, int cand,
        double fb_temp, bool novel_only, double qidf_alpha = 0.0,
        double k1_round2 = -1.0, double b_round2 = -1.0,
        double fb_dl_exp = 1.0, double qtf_exp = 1.0,
        double fb_idf_exp = 0.0, double fb_tf_exp = 1.0,
        double fb_vsm_weight = 0.0) {
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
            for (auto &kv : qc) {
                double qweight = std::pow(static_cast<double>(kv.second), qtf_exp);
                if (qidf_alpha != 0.0) {
                    double idf = std::log(
                        (static_cast<double>(N) - t_df[kv.first] + 0.5)
                        / (t_df[kv.first] + 0.5) + 1.0);
                    qweight *= std::pow(std::max(idf, 1e-12), qidf_alpha);
                }
                q0.push_back({kv.first, qweight});
            }
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
        if (fb_vsm_weight <= 0.0) {
            std::partial_sort(items.begin(), items.begin() + RR, items.end(), by_score);
        } else {
            // Use the top candidate-depth BM25 docs and mix their BM25 and
            // original-query VSM ranks for pseudo-relevance selection.
            size_t DD = std::min(static_cast<size_t>(cand), items.size());
            std::partial_sort(items.begin(), items.begin() + DD, items.end(), by_score);
            std::unordered_map<int, int> vrank;
            auto vhits = vsm(tokens, cand);
            for (size_t i = 0; i < vhits.size(); i++) {
                auto dit = doc_ord.find(vhits[i].first);
                if (dit != doc_ord.end()) vrank[dit->second] = static_cast<int>(i + 1);
            }
            std::vector<std::pair<double, int>> mixed;
            mixed.reserve(DD);
            for (size_t i = 0; i < DD; i++) {
                int d = items[i].second;
                double br = 1.0 / (60.0 + static_cast<double>(i + 1));
                auto it = vrank.find(d);
                double vr = it == vrank.end()
                    ? 0.0
                    : 1.0 / (60.0 + static_cast<double>(it->second));
                mixed.push_back({(1.0 - fb_vsm_weight) * br +
                                 fb_vsm_weight * vr, d});
            }
            std::partial_sort(mixed.begin(), mixed.begin() + RR, mixed.end(), by_score);
            items.swap(mixed);
        }
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
                double term_idf = std::log(
                    static_cast<double>(N) / t_df[tw]);
                rel[tw] += pd * (std::pow(static_cast<double>(fwd_tfs[j]), fb_tf_exp) /
                                  std::pow(static_cast<double>(dl), fb_dl_exp)) *
                           std::pow(std::max(term_idf, 1e-12), fb_idf_exp);
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
        return topk(score, touched, cand);
    }

    // Run the two fusion arms (RM3-or-BM25 and VSM) concurrently. Both arms
    // only read shared, immutable arrays (t_ords/t_tfs/doc_len/doc_norm/fwd_*)
    // and each owns its score/touched buffers, so concurrent reads are safe
    // with no locking. The VSM arm runs on a worker thread while this thread
    // runs the (heavier) RM3/BM25 arm. Called with the GIL released (see the
    // pybind def), so the two C++ threads run truly in parallel. Fusion (RRF)
    // stays in Python. Returns {bm_hits, vs_hits}.
    std::pair<std::vector<std::pair<std::string, double>>,
              std::vector<std::pair<std::string, double>>>
    arms(const std::vector<std::string> &tokens, bool use_rm3,
         int R, int M, double lambda_, double k1, double b, int cand,
         double fb_temp, bool novel_only,
         double k1_round2 = -1.0, double b_round2 = -1.0) {
        std::vector<std::pair<std::string, double>> vs_hits;
        std::thread vt([&] { vs_hits = vsm(tokens, cand); });
        std::vector<std::pair<std::string, double>> bm_hits =
            (use_rm3 && has_forward)
                ? rm3(tokens, R, M, lambda_, k1, b, cand, fb_temp, novel_only,
                     0.0, k1_round2, b_round2)
                : bm25(tokens, k1, b, cand);
        vt.join();
        return {bm_hits, vs_hits};
    }
};

PYBIND11_MODULE(_index_cpp, m) {
    m.doc() = "Native C++ inverted index: BM25 + VSM over the compressed postings.";
    py::class_<NativeIndex>(m, "NativeIndex")
        .def(py::init<const std::string &>())
        .def("bm25", &NativeIndex::bm25,
             py::arg("tokens"), py::arg("k1"), py::arg("b"), py::arg("k"))
        .def("bm25plus", &NativeIndex::bm25plus,
             py::arg("tokens"), py::arg("k1"), py::arg("b"), py::arg("k"),
             py::arg("delta"))
        .def("bm25l", &NativeIndex::bm25l,
             py::arg("tokens"), py::arg("k1"), py::arg("b"), py::arg("k"))
        .def("coverage", &NativeIndex::coverage,
             py::arg("tokens"), py::arg("k"), py::arg("idf_weighted"),
             py::arg("idf_power") = 1.0)
        .def("early_match", &NativeIndex::early_match,
             py::arg("tokens"), py::arg("k"))
        .def("document_length", &NativeIndex::document_length,
             py::arg("doc_id"))
        .def("vsm", &NativeIndex::vsm, py::arg("tokens"), py::arg("k"))
        .def("rm3", &NativeIndex::rm3,
             py::arg("tokens"), py::arg("R"), py::arg("M"), py::arg("lambda_"),
             py::arg("k1"), py::arg("b"), py::arg("cand"),
             py::arg("fb_temp"), py::arg("novel_only"),
             py::arg("qidf_alpha") = 0.0,
             py::arg("k1_round2") = -1.0,
             py::arg("b_round2") = -1.0,
             py::arg("fb_dl_exp") = 1.0,
             py::arg("qtf_exp") = 1.0,
             py::arg("fb_idf_exp") = 0.0,
             py::arg("fb_tf_exp") = 1.0,
             py::arg("fb_vsm_weight") = 0.0)
        .def("arms", &NativeIndex::arms,
             py::arg("tokens"), py::arg("use_rm3"), py::arg("R"), py::arg("M"),
             py::arg("lambda_"), py::arg("k1"), py::arg("b"), py::arg("cand"),
             py::arg("fb_temp"), py::arg("novel_only"),
             py::arg("k1_round2") = -1.0,
             py::arg("b_round2") = -1.0,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("has_forward",
                               [](const NativeIndex &n) { return n.has_forward; });
}
