"""
scripts/tune_bm25.py — sweep BM25's k1 and b on the dev set and plot the result.

Methodology (kept deliberately simple):

  1. Build the inverted index ONCE. The tokenizer (v1 + df-ratio gate) is
     held fixed; only k1 and b vary. So every grid point is scored against
     the exact same postings — the sweep measures the effect of k1/b alone.

  2. Grid: a coarse, canonical-centred grid (k1 in [0.8..2.5], b in
     [0.3..0.9]). Coarse on purpose — with only 50 dev queries, a fine
     grid just fits noise between neighbouring points.

  3. Metric: harness.metrics.ndcg_at_k, the identical function the
     leaderboard uses, so these numbers match grading. Each grid point's
     score is the mean nDCG@10 over the 50 dev queries.

  4. Output: the full grid is printed and saved (JSON + CSV), and a plot
     of nDCG@10 vs k1 (one line per b value) is written to a PNG for the
     report.

Choosing k1/b from this is a judgement call, NOT a raw argmax: the surface
climbs toward the grid edge (high k1, low b), but the top is a broad,
flat plateau and the edge is exactly where overfitting to 50 queries
lives. We pick a robust interior point on the plateau near BM25's
canonical range (k1~1.5-1.8, b~0.6-0.75) rather than chasing the fragile
maximum. The plot makes that plateau visible.

Usage:
    python -m scripts.tune_bm25 \
        --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv \
        --qrels data/full/qrels_dev.txt \
        --out-json runs/bm25_sweep.json \
        --out-plot runs/bm25_sweep.png
"""

import argparse
import json
import math

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, tokenize
from harness.metrics import ndcg_at_k

K1_GRID = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
B_GRID = [0.3, 0.5, 0.6, 0.75, 0.9]


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


def bm25_topk(index, q_tokens, k1, b, k=10):
    """BM25 top-k doc_ids for pre-tokenised query terms, best first."""
    scores = {}
    for tok in q_tokens:
        postings = index.postings.get(tok)
        if not postings:
            continue
        df = len(postings)
        idf = math.log((index.N - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in postings.items():
            dl = index.doc_len[doc_id]
            denom = tf + k1 * (1 - b + b * dl / index.avg_doc_len)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1)) / denom
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [d for d, _ in ranked]


def make_plot(grid, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for b in B_GRID:
        ys = [grid[(k1, b)] for k1 in K1_GRID]
        ax.plot(K1_GRID, ys, marker="o", label=f"b = {b}")
    ax.set_xlabel("k1")
    ax.set_ylabel("nDCG@10 (dev, 50 queries)")
    ax.set_title("BM25 parameter sweep")
    ax.legend(title="length norm.")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out-json", default="runs/bm25_sweep.json")
    ap.add_argument("--out-plot", default="runs/bm25_sweep.png")
    args = ap.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} docs. Building index once...")
    index = InvertedIndex()
    index.build(corpus)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    qids = [q for q, _ in queries if q in qrels]
    q_tokens = {q: tokenize(text) for q, text in queries}

    # Score the whole grid.
    grid = {}  # (k1, b) -> mean nDCG@10
    for k1 in K1_GRID:
        for b in B_GRID:
            total = 0.0
            for q in qids:
                ranked = bm25_topk(index, q_tokens[q], k1, b)
                total += ndcg_at_k(ranked, qrels[q], k=10)
            grid[(k1, b)] = total / len(qids)

    # Print grid (rows=k1, cols=b).
    print("\nnDCG@10  (rows = k1, cols = b)")
    print("  k1\\b " + "".join(f"{b:>8}" for b in B_GRID))
    for k1 in K1_GRID:
        print(f"{k1:>6} " + "".join(f"{grid[(k1, b)]:>8.4f}" for b in B_GRID))

    best = max(grid, key=grid.get)
    print(f"\nGrid maximum: k1={best[0]}, b={best[1]}  (nDCG@10 {grid[best]:.4f})")
    print("(See docstring: we ship a robust interior point, not this edge max.)")

    # Save results (JSON + CSV) for the report.
    with open(args.out_json, "w") as f:
        json.dump({f"{k1},{b}": v for (k1, b), v in grid.items()}, f, indent=2)
    print(f"Wrote {args.out_json}")

    csv_path = args.out_json.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("k1,b,ndcg@10\n")
        for (k1, b), v in grid.items():
            f.write(f"{k1},{b},{v:.6f}\n")
    print(f"Wrote {csv_path}")

    make_plot(grid, args.out_plot)


if __name__ == "__main__":
    main()
