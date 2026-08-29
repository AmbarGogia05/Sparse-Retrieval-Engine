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
#include <iterator>
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

// Read one VByte from a byte buffer, advancing pos (mirror of put_vbyte).
static long get_vbyte(const unsigned char *p, size_t &pos) {
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

template <typename T>
static void wr(std::ofstream &f, T v) { f.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
template <typename T>
static T rd(std::ifstream &f) { T v; f.read(reinterpret_cast<char *>(&v), sizeof(T)); return v; }

// --- A1 codec: fold "tf == 1" into the gap ---------------------------------
// A posting list (ascending absolute ords, parallel tfs) is written as: df
// codes (one per posting, `gap*2 + (tf>1 ? 1 : 0)`), followed by ONLY the
// tfs for postings whose flag bit is set (tf==1 postings emit no tf byte at
// all — 73.7% of postings in this corpus, so this is the bulk of the size
// win). Shared by postings.bin (per-term, ords = doc ordinals) and
// forward.bin (per-doc, ords = term ordinals) — same shape of problem, same
// codec, same gap-then-conditional-tfs layout as the pre-A1 format so the
// change is a pure re-encoding.
static void put_postings(std::string &out, const std::vector<uint32_t> &ords,
                          const std::vector<uint32_t> &tfs) {
    uint32_t prev = 0;
    for (size_t i = 0; i < ords.size(); i++) {
        uint32_t gap = (i == 0) ? ords[i] : ords[i] - prev;
        prev = ords[i];
        uint32_t code = (gap << 1) | (tfs[i] > 1 ? 1u : 0u);
        put_vbyte(out, static_cast<long>(code));
    }
    for (size_t i = 0; i < ords.size(); i++)
        if (tfs[i] > 1) put_vbyte(out, static_cast<long>(tfs[i]));
}

// Decode one worker's forward.tmp (local-id gaps + tfs per doc, written by
// SpimiBuilder::add_document) and re-encode it with local ids remapped to
// final (global) term ordinals via local_to_final, dropping pruned terms
// (local_to_final[id] == -1) and re-sorting each doc's terms by final
// ordinal. Shared by SpimiBuilder::write_forward (serial path) and
// finalize_parallel (parallel path) so both produce byte-identical output
// for the same (forward.tmp, n_docs, local_to_final) input.
static std::string encode_forward_bytes(const std::string &forward_tmp_path, uint32_t n_docs,
                                         const std::vector<int> &local_to_final) {
    std::ifstream f(forward_tmp_path, std::ios::binary);
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    const unsigned char *p = reinterpret_cast<const unsigned char *>(buf.data());
    size_t pos = 0;
    std::string out;
    for (uint32_t d = 0; d < n_docs; d++) {
        long n = get_vbyte(p, pos);
        std::vector<uint32_t> locs(n);
        std::vector<uint32_t> tfs(n);
        long prev = 0;
        for (long i = 0; i < n; i++) {
            long g = get_vbyte(p, pos);
            prev = (i == 0) ? g : prev + g;
            locs[i] = static_cast<uint32_t>(prev);
        }
        for (long i = 0; i < n; i++) tfs[i] = static_cast<uint32_t>(get_vbyte(p, pos));

        std::vector<std::pair<int, uint32_t>> keep;  // (final ord, tf)
        keep.reserve(n);
        for (long i = 0; i < n; i++) {
            int fo = local_to_final[locs[i]];
            if (fo >= 0) keep.push_back({fo, tfs[i]});
        }
        std::sort(keep.begin(), keep.end());

        put_vbyte(out, static_cast<long>(keep.size()));
        std::vector<uint32_t> kord, ktf;
        kord.reserve(keep.size());
        ktf.reserve(keep.size());
        for (auto &kv : keep) { kord.push_back(static_cast<uint32_t>(kv.first)); ktf.push_back(kv.second); }
        put_postings(out, kord, ktf);
    }
    return out;
}

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
    uint32_t n_docs = 0;      // next doc gets this ordinal (starts at start_ord)
    uint32_t local_docs = 0;  // number of add_document() calls on THIS instance
    long total_len = 0;
    std::vector<std::string> block_files;
    std::ofstream docs_out;
    // A4: doc lengths move out of docs.txt (which now holds ONLY verbatim
    // doc-id lines) into their own VByte stream, written in lockstep with
    // docs_out so row i of docs.txt and the i-th VByte in doclen_out always
    // refer to the same doc.
    std::ofstream doclen_out;
    std::unordered_set<std::string> case_terms_seen;

    // Forward index (doc -> its terms+tfs), needed to build RM3 relevance
    // models at query time. Term ordinals aren't known until the final merge
    // fixes the vocabulary order, so during streaming we write each doc's
    // vector keyed by a stable *local* id, then remap local -> final ord in a
    // single pass in finalize(). `vocab` (string -> local id) is bounded by
    // vocabulary size (~10^5), not by the postings, so it doesn't threaten the
    // OOM-safety the block flushing provides.
    std::unordered_map<std::string, uint32_t> vocab;
    std::ofstream fwd_tmp;

    SpimiBuilder(const std::string &d, size_t threshold, uint32_t start_ord = 0)
        : dir(d), flush_threshold(threshold), n_docs(start_ord), docs_out(d + "/docs.txt"),
          doclen_out(d + "/doclen.bin", std::ios::binary),
          fwd_tmp(d + "/forward.tmp", std::ios::binary) {}

    uint32_t local_id(const std::string &t) {
        auto it = vocab.find(t);
        if (it != vocab.end()) return it->second;
        uint32_t id = static_cast<uint32_t>(vocab.size());
        vocab.emplace(t, id);
        return id;
    }

    static bool has_upper(const std::string &s) {
        for (char c : s)
            if (c >= 'A' && c <= 'Z') return true;
        return false;
    }

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
        for (const auto &t : tokens) {
            tf[t]++;
            if (has_upper(t)) case_terms_seen.insert(t);
        }
        std::vector<std::pair<uint32_t, uint32_t>> fwd;  // (local id, tf)
        fwd.reserve(tf.size());
        for (auto &kv : tf) {
            auto &vec = block[kv.first];
            vec.push_back({ord, kv.second});
            cur_postings++;
            fwd.push_back({local_id(kv.first), kv.second});
        }
        // Append this doc's forward vector (local-id gaps + tfs) to forward.tmp.
        std::sort(fwd.begin(), fwd.end());
        std::string rec;
        put_vbyte(rec, static_cast<long>(fwd.size()));
        uint32_t prev = 0;
        for (size_t i = 0; i < fwd.size(); i++) {
            put_vbyte(rec, i == 0 ? fwd[i].first : fwd[i].first - prev);
            prev = fwd[i].first;
        }
        for (auto &p : fwd) put_vbyte(rec, p.second);
        fwd_tmp.write(rec.data(), rec.size());

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

    // --- Parallel-build worker API ---------------------------------------
    // Used when a SpimiBuilder is one of N per-worker builders (each owning
    // a contiguous, globally-ordinalled slice of the corpus) instead of the
    // sole builder for the whole corpus. close_worker() flushes the last
    // in-memory block and closes this worker's docs.txt/forward.tmp files
    // without doing any cross-worker merge work; the parent process then
    // hands the accessors below to finalize_parallel() to do the k-way
    // merge (see finalize_parallel below for how ord/df-hash-order identity
    // with the serial path is preserved).
    void close_worker() {
        flush_block();
        docs_out.close();
        doclen_out.close();
        fwd_tmp.close();
    }

    std::vector<std::string> get_block_files() const { return block_files; }
    std::unordered_map<std::string, uint32_t> get_vocab() const { return vocab; }
    // NOTE: get_n_docs() (below) is the running *global ordinal* counter
    // (n_docs == start_ord + docs added so far); get_local_docs() is the
    // count of add_document() calls on this instance. They coincide when
    // start_ord == 0 (the serial-path / single-builder case), but differ
    // for a parallel worker with a nonzero start_ord — using the wrong one
    // for the forward-index row count or the cross-worker N summation is
    // an out-of-bounds read (this bit me once during development).
    uint32_t get_n_docs() const { return n_docs; }
    uint32_t get_local_docs() const { return local_docs; }
    long get_total_len() const { return total_len; }
    std::string docs_path() const { return dir + "/docs.txt"; }
    std::string doclen_path() const { return dir + "/doclen.bin"; }
    std::string forward_tmp_path() const { return dir + "/forward.tmp"; }

    // Second pass over forward.tmp: remap local ids -> final term ordinals,
    // drop pruned terms, re-sort each doc's terms by final ordinal, and write
    // the compressed forward.bin (per doc: VByte count, then final-ordinal
    // gaps, then tfs). Doc order matches docs.txt (streamed in add order).
    void write_forward(const std::vector<int> &local_to_final) {
        std::string out = encode_forward_bytes(dir + "/forward.tmp", n_docs, local_to_final);
        std::ofstream(dir + "/forward.bin", std::ios::binary).write(out.data(), out.size());
    }

    void finalize(const std::unordered_map<std::string, std::string> &canonical, double df_ratio) {
        flush_block();
        docs_out.close();
        doclen_out.close();
        fwd_tmp.close();

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

        // A2: final term ordinals are assigned by DESCENDING global df (ties
        // broken by term string, ascending, for determinism), not by the
        // alphabetical order the merge below happens to visit terms in. This
        // clusters each document's ~67 distinct terms at low ordinals in the
        // forward index (mean gap ~4100 -> much smaller), at the cost of one
        // extra pass. Pruned terms never get an ordinal.
        std::vector<std::string> kept_terms;
        kept_terms.reserve(df_all.size());
        for (auto &kv : df_all)
            if (!prune.count(kv.first)) kept_terms.push_back(kv.first);
        std::sort(kept_terms.begin(), kept_terms.end(), [&](const std::string &a, const std::string &b) {
            uint32_t da = df_all[a], db = df_all[b];
            if (da != db) return da > db;
            return a < b;
        });
        std::unordered_map<std::string, int> final_ord_map;
        final_ord_map.reserve(kept_terms.size() * 2);
        for (size_t i = 0; i < kept_terms.size(); i++) final_ord_map[kept_terms[i]] = static_cast<int>(i);

        // Pass 2: k-way merge by term (alphabetical, cheap with BlockReader),
        // buffering each surviving term's postings at its FINAL (df-order)
        // slot so they can be written out in final-ordinal order afterward.
        std::vector<BlockReader *> readers;
        for (auto &path : block_files) readers.push_back(new BlockReader(path));

        struct PerTerm { std::vector<uint32_t> ords, tfs; };
        std::vector<PerTerm> data(kept_terms.size());

        // Map each surviving term's local id -> its final ordinal (position in
        // terms.txt). Pruned / never-emitted terms keep -1 so the forward-index
        // remap drops them. NativeIndex assigns term ids in terms.txt order, so
        // "final ordinal" == the id RM3 will look terms up by.
        std::vector<int> local_to_final(vocab.size(), -1);

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

            int fo = final_ord_map[mterm];
            local_to_final[vocab[mterm]] = fo;
            data[fo].ords = std::move(ords);
            data[fo].tfs = std::move(tfs);
        }
        for (auto *r : readers) delete r;

        std::ofstream terms_out(dir + "/terms.txt");
        std::string postings;  // buffered, written once
        for (size_t i = 0; i < kept_terms.size(); i++) {
            if (i) terms_out << '\n';
            terms_out << kept_terms[i] << '\t' << data[i].ords.size();
            put_postings(postings, data[i].ords, data[i].tfs);  // A1 codec
        }
        std::ofstream(dir + "/postings.bin", std::ios::binary).write(postings.data(), postings.size());

        // forward.bin is intentionally NOT written: it is a byte-for-byte
        // redundant transpose of postings.bin, and NativeIndex reconstructs the
        // forward index in RAM at load time (see _index_cpp.cpp load()). This
        // roughly halves the persisted index size. write_forward()/
        // encode_forward_bytes() are kept for reference but no longer called.

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
        std::remove((dir + "/forward.tmp").c_str());
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
// 2. df_all/keep_case (meta.json case_terms) insertion order: unordered_set
//    iteration order depends on insertion order for keys that collide into
//    the same hash bucket, so a naive "read blocks in worker order" Pass 1
//    would build df_all with different bucket contents than the serial
//    path and could emit case_terms in a different order — same set,
//    different bytes. The serial path's single block (this corpus never
//    triggers a mid-stream flush) is written by flush_block() in global
//    alphabetical term order, so its Pass 1 inserts into df_all in that
//    exact order. We replicate that here by first aggregating every
//    worker's block headers into a std::map (alphabetically ordered by
//    construction, regardless of read order) and then inserting into the
//    unordered_map df_all in that alphabetical order — reproducing the
//    same df_all bucket layout, hence the same downstream keep_case
//    insertion order and the same case_terms byte output as the serial
//    build.
void finalize_parallel(const py::list &workers, const std::string &out_dir,
                        const std::unordered_map<std::string, std::string> &canonical,
                        double df_ratio) {
    struct WorkerData {
        std::vector<std::string> block_files;
        std::unordered_map<std::string, uint32_t> vocab;
        std::string docs_txt;
        std::string doclen_bin;
        std::string forward_tmp;
        uint32_t n_docs;  // LOCAL doc count for this worker (not a global
                           // ordinal) — see SpimiBuilder::get_local_docs.
        long total_len;
    };

    std::vector<WorkerData> ws;
    for (auto &item : workers) {
        py::dict d = item.cast<py::dict>();
        WorkerData w;
        w.block_files = d["block_files"].cast<std::vector<std::string>>();
        w.vocab = d["vocab"].cast<std::unordered_map<std::string, uint32_t>>();
        w.docs_txt = d["docs_txt"].cast<std::string>();
        w.doclen_bin = d["doclen_bin"].cast<std::string>();
        w.forward_tmp = d["forward_tmp"].cast<std::string>();
        w.n_docs = d["n_docs"].cast<uint32_t>();
        w.total_len = d["total_len"].cast<long>();
        ws.push_back(std::move(w));
    }

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

    // Pass 1a: aggregate df per term into a *sorted* map first (order of
    // reads doesn't matter, std::map is always alphabetical), then...
    std::map<std::string, uint32_t> tmp_df;
    for (auto &path : all_blocks) {
        std::ifstream f(path, std::ios::binary);
        uint32_t tl;
        while (f.read(reinterpret_cast<char *>(&tl), sizeof(tl))) {
            std::string term(tl, '\0');
            f.read(&term[0], tl);
            uint32_t df = rd<uint32_t>(f);
            tmp_df[term] += df;
            f.seekg(static_cast<std::streamoff>(df) * 2 * sizeof(uint32_t), std::ios::cur);
        }
    }
    // ...Pass 1b: insert into the unordered_map in that alphabetical order,
    // matching the serial path's single-sorted-block Pass 1 exactly.
    std::unordered_map<std::string, uint32_t> df_all;
    for (auto &kv : tmp_df) df_all[kv.first] = kv.second;

    // Truecase gate (identical logic/order to SpimiBuilder::finalize).
    std::unordered_set<std::string> prune, keep_case;
    for (auto &kv : df_all) {
        const std::string &term = kv.first;
        if (!SpimiBuilder::has_upper(term)) continue;
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

    // A2 (same as the serial path): final ordinals by descending global df,
    // ties broken by term string ascending — NOT by the alphabetical merge
    // order below, and NOT by which worker happened to see the term first.
    std::vector<std::string> kept_terms;
    kept_terms.reserve(df_all.size());
    for (auto &kv : df_all)
        if (!prune.count(kv.first)) kept_terms.push_back(kv.first);
    std::sort(kept_terms.begin(), kept_terms.end(), [&](const std::string &a, const std::string &b) {
        uint32_t da = df_all[a], db = df_all[b];
        if (da != db) return da > db;
        return a < b;
    });
    std::unordered_map<std::string, int> final_ord_map;
    final_ord_map.reserve(kept_terms.size() * 2);
    for (size_t i = 0; i < kept_terms.size(); i++) final_ord_map[kept_terms[i]] = static_cast<int>(i);

    // Pass 2: k-way merge, readers ordered [worker0 blocks..., worker1
    // blocks..., ...] so per-term postings concatenate in ascending
    // (global) ordinal order without any remap. Buffer each surviving
    // term's postings at its final (df-order) slot, same as the serial path.
    std::vector<BlockReader *> readers;
    for (auto &w : ws)
        for (auto &p : w.block_files) readers.push_back(new BlockReader(p));

    struct PerTerm { std::vector<uint32_t> ords, tfs; };
    std::vector<PerTerm> data(kept_terms.size());

    std::vector<std::vector<int>> local_to_final(ws.size());
    for (size_t i = 0; i < ws.size(); i++) local_to_final[i].assign(ws[i].vocab.size(), -1);

    while (true) {
        std::string mterm;
        bool found = false;
        for (auto *r : readers)
            if (r->ok && (!found || r->term < mterm)) { mterm = r->term; found = true; }
        if (!found) break;

        std::vector<uint32_t> ords, tfs;
        for (auto *r : readers) {
            if (r->ok && r->term == mterm) {
                ords.insert(ords.end(), r->ords.begin(), r->ords.end());
                tfs.insert(tfs.end(), r->tfs.begin(), r->tfs.end());
                r->advance();
            }
        }
        if (prune.count(mterm)) continue;

        int fo = final_ord_map[mterm];
        for (size_t wi = 0; wi < ws.size(); wi++) {
            auto it = ws[wi].vocab.find(mterm);
            if (it != ws[wi].vocab.end()) local_to_final[wi][it->second] = fo;
        }
        data[fo].ords = std::move(ords);
        data[fo].tfs = std::move(tfs);
    }
    for (auto *r : readers) delete r;

    std::ofstream terms_out(out_dir + "/terms.txt");
    std::string postings;
    for (size_t i = 0; i < kept_terms.size(); i++) {
        if (i) terms_out << '\n';
        terms_out << kept_terms[i] << '\t' << data[i].ords.size();
        put_postings(postings, data[i].ords, data[i].tfs);  // A1 codec
    }
    std::ofstream(out_dir + "/postings.bin", std::ios::binary).write(postings.data(), postings.size());

    // forward.bin is intentionally NOT written here (nor in the serial path):
    // it is a redundant transpose of postings.bin and is reconstructed in RAM
    // at load time by NativeIndex (see _index_cpp.cpp load()). The per-worker
    // forward.tmp scratch is still produced and removed below; only the final
    // persisted forward.bin is dropped.

    // meta.json — identical format/logic to SpimiBuilder::finalize.
    double avg = N ? static_cast<double>(total_len) / N : 0.0;
    std::ofstream meta(out_dir + "/meta.json");
    meta << std::setprecision(std::numeric_limits<double>::max_digits10);
    meta << "{\"N\": " << N << ", \"avg_doc_len\": " << avg << ", \"case_terms\": [";
    bool f2 = true;
    for (auto &t : keep_case) {
        if (!f2) meta << ", ";
        f2 = false;
        meta << '"' << t << '"';
    }
    meta << "]}";
    meta.close();

    for (auto &path : all_blocks) std::remove(path.c_str());
    for (auto &w : ws) {
        std::remove(w.forward_tmp.c_str());
        std::remove(w.docs_txt.c_str());    // already concatenated into out_dir/docs.txt above
        std::remove(w.doclen_bin.c_str());  // already concatenated into out_dir/doclen.bin above
    }
}

PYBIND11_MODULE(_spimi_cpp, m) {
    m.doc() = "SPIMI index builder: block-flushing, writes save_v2's format.";
    py::class_<SpimiBuilder>(m, "SpimiBuilder")
        .def(py::init<const std::string &, size_t, uint32_t>(),
             py::arg("index_dir"), py::arg("flush_threshold"), py::arg("start_ord") = 0)
        .def("add_document", &SpimiBuilder::add_document,
             py::arg("doc_id"), py::arg("tokens"), py::arg("doc_len"))
        .def("case_terms", &SpimiBuilder::case_terms)
        .def("finalize", &SpimiBuilder::finalize, py::arg("canonical"), py::arg("df_ratio"))
        .def("close_worker", &SpimiBuilder::close_worker)
        .def("block_files", &SpimiBuilder::get_block_files)
        .def("vocab", &SpimiBuilder::get_vocab)
        .def("n_docs_count", &SpimiBuilder::get_local_docs)
        .def("total_len_count", &SpimiBuilder::get_total_len)
        .def("docs_path", &SpimiBuilder::docs_path)
        .def("doclen_path", &SpimiBuilder::doclen_path)
        .def("forward_tmp_path", &SpimiBuilder::forward_tmp_path);
    m.def("finalize_parallel", &finalize_parallel,
          py::arg("workers"), py::arg("out_dir"), py::arg("canonical"), py::arg("df_ratio"));
}
