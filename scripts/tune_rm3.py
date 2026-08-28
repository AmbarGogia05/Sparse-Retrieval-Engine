"""
scripts/tune_rm3.py — sweep RM3 pseudo-relevance-feedback params on the dev set.

RM3 in one line: run BM25, treat the top-R docs as relevant, build a relevance
model over their terms, mix it with the original query (weight lambda), and
re-run a *weighted* BM25 with the expanded query. See _index_cpp.NativeIndex.rm3.

Methodology (mirrors tune_bm25.py, and the same overfitting discipline):

  1. Build the index ONCE (production path: SPIMI + forward index), load the
     native scorer. Every grid point scores the same postings.

  2. Fixed: R = 10 feedback docs (per design), k1/b = 1.8/0.6 (shipped BM25),
     and the fusion (RM3-BM25 fused with VSM on the *original* query, weighted RRF
     W_BM=0.6, exactly the shipped fusion with BM25 swapped for RM3-BM25).

  3. Swept: M (number of expansion terms) x lambda (anchor weight on the
     original query; lambda=1.0 recovers plain BM25, i.e. no expansion).

  4. Metric: harness.metrics.ndcg_at_k (the leaderboard's own function), mean
     over the 50 dev queries. We ALWAYS score through the fusion, never RM3
     alone — a change only counts if it survives in the shipped pipeline.

Read the plot as a plateau, not an argmax: with 50 queries the peak grid cell
is noisy. Ship a robust interior (M, lambda), and if the whole surface sits at
or below the lambda=1.0 row, RM3 doesn't help and we don't ship it.

Usage:
    python -m scripts.tune_rm3 \
        --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv \
        --qrels data/full/qrels_dev.txt \
        --out-json runs/rm3_sweep.json \
        --out-plot runs/rm3_sweep.png
"""

import argparse
import json
import os
import shutil
import tempfile

from submission import retrieve as R
from submission.indexer import tokenize, set_case_terms
from harness.metrics import ndcg_at_k

# Fixed knobs (see docstring).
R_FEEDBACK = 10
BM25_K1 = 1.8
BM25_B = 0.6
CAND = 1000
RRF_K = 60
W_BM = 0.6

# Swept grids.
M_GRID = [5, 10, 15, 20, 30]
LAMBDA_GRID = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # 1.0 == plain BM25

# 20 common English function words for the expansion stoplist, stemmed so they
# match indexed terms. Deliberately NOT topical (no "covid"/"19") — a frequent
# topical term has real IDF, a function word does not.


def read_queries(path):
    out = []
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            out.append((qid, text))
    return out


def read_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, doc_id, rel = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def fuse(bm_hits, vs_hits, k=10):
    """Weighted RRF of two (doc_id, score) lists — identical to custom_scorer."""
    def ranks(hits):
        return {d: i for i, (d, _) in enumerate(hits, start=1)}
    bm, vs = ranks(bm_hits), ranks(vs_hits)
    fused = {}
    for d, r in bm.items():
        fused[d] = fused.get(d, 0.0) + W_BM / (RRF_K + r)
    for d, r in vs.items():
        fused[d] = fused.get(d, 0.0) + (1 - W_BM) / (RRF_K + r)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


def make_plot(grid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in LAMBDA_GRID:
        ys = [grid[(m, lam)] for m in M_GRID]
        label = "lambda = 1.0 (BM25)" if lam == 1.0 else f"lambda = {lam}"
        ax.plot(M_GRID, ys, marker="o", label=label)
    ax.set_xlabel("M (expansion terms)")
    ax.set_ylabel("nDCG@10 (dev, fused with VSM)")
    ax.set_title(f"RM3 sweep (R={R_FEEDBACK}, fused RRF W_BM={W_BM})")
    ax.legend(title="anchor weight", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out-json", default="runs/rm3_sweep.json")
    ap.add_argument("--out-plot", default="runs/rm3_sweep.png")
    ap.add_argument("--index-dir", default=None,
                    help="reuse an existing built index instead of building one")
    args = ap.parse_args()

    tmp = None
    index_dir = args.index_dir
    if index_dir is None:
        tmp = tempfile.mkdtemp(prefix="rm3_idx_")
        index_dir = tmp
        print(f"Building index once (production path) into {index_dir} ...")
        R.build_index(args.corpus, index_dir)

    # Load the native index directly (we need rm3()/vsm(), not retrieve()).
    from submission._index_cpp import NativeIndex
    idx = NativeIndex(index_dir)
    if not idx.has_forward:
        raise SystemExit("forward.bin missing/empty — RM3 needs the forward index.")

    # Query-side case filtering must match the build (set_case_terms from meta).
    with open(os.path.join(index_dir, "meta.json")) as f:
        set_case_terms(set(json.load(f).get("case_terms", [])))


    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    qids = [q for q, _ in queries if q in qrels]
    q_tokens = {q: tokenize(text) for q, text in queries}
    # VSM on the original query is identical across the grid — compute once.
    vs_cache = {q: idx.vsm(q_tokens[q], CAND) for q in qids}

    grid = {}  # (M, lambda) -> mean nDCG@10
    for m in M_GRID:
        for lam in LAMBDA_GRID:
            total = 0.0
            for q in qids:
                bm = idx.rm3(q_tokens[q], R_FEEDBACK, m, lam, BM25_K1, BM25_B, CAND)
                total += ndcg_at_k([d for d, _ in fuse(bm, vs_cache[q])], qrels[q], k=10)
            grid[(m, lam)] = total / len(qids)

    print("\nnDCG@10  (rows = M, cols = lambda)")
    print("   M\\lam " + "".join(f"{lam:>8}" for lam in LAMBDA_GRID))
    for m in M_GRID:
        print(f"{m:>6}  " + "".join(f"{grid[(m, lam)]:>8.4f}" for lam in LAMBDA_GRID))

    base = max(grid[(m, 1.0)] for m in M_GRID)  # lambda=1.0 == plain BM25
    best = max(grid, key=grid.get)
    print(f"\nBaseline (lambda=1.0, plain BM25 fused): {base:.4f}")
    print(f"Grid maximum: M={best[0]}, lambda={best[1]}  (nDCG@10 {grid[best]:.4f}, "
          f"{grid[best] - base:+.4f} vs baseline)")
    print("(Ship a robust interior point, not this edge max — see docstring.)")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({f"{m},{lam}": v for (m, lam), v in grid.items()}, f, indent=2)
    print(f"Wrote {args.out_json}")
    make_plot(grid, args.out_plot)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
