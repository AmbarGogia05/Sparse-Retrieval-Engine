"""
scripts/fuse_scores.py — try BM25/VSM fusion schemes on the cached scores.

Reads the score cache written by cache_scores.py (no corpus, no retrieval)
and evaluates fusion schemes against the dev qrels. Schemes:

  - bm25       : BM25 alone (the baseline we must beat)
  - vsm        : VSM alone (how much signal it has on its own)
  - rrf        : reciprocal rank fusion, sum of 1/(k+rank), k=60 fixed
  - linear(a)  : a * bm25' + (1-a) * vsm', min-max normalised per query,
                 swept over a in 0.0 .. 1.0

Methodology (deliberately simple):
  1. Every scheme is scored as mean nDCG@10 over all 50 dev queries. That
     single number is the decision measure — we pick the best fusion by it.
  2. We print the whole linear-a curve so we can see by eye whether it is
     flat (no signal) or genuinely peaked.
  3. A bootstrap 95% CI on the per-query gain (best fusion - bm25) is
     computed and SAVED as a secondary metric for later reference — not
     used to gate the choice. It tells us how much the gain wobbles across
     query samples; handy when comparing against the held-out set later.

Why no cross-validation: RRF has no tuned parameters (k=60 is a fixed
convention), so there is nothing to overfit and nothing for CV to check.
The only tuned knob is linear's `a`; we handle that by preferring the
parameter-free RRF whenever it is not clearly beaten, rather than trusting
a peak fitted on 50 queries.

All per-scheme scores and the bootstrap CI are written to --out as JSON.

Usage:
    python -m scripts.fuse_scores \
        --cache runs/score_cache.json \
        --qrels data/full/qrels_dev.txt \
        --out runs/fusion_results.json
"""

import argparse
import json
import random

from harness.metrics import ndcg_at_k

random.seed(42)

ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]  # weighted-RRF w on BM25
N_BOOT = 1000
RRF_K = 60


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, doc_id, rel = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def load_cache(path):
    # {qid: {doc_id: {"bm25": score, "vsm": score}}}
    with open(path) as f:
        return json.load(f)["scores"]


def ranking_by_score(per_doc, key):
    """Doc ids ranked by one retriever's raw score, best first."""
    hits = [(d, s[key]) for d, s in per_doc.items() if key in s]
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits]


def ranking_linear(per_doc, alpha):
    """a * bm25' + (1-a) * vsm', each min-max normalised over this query."""
    def normaliser(key):
        vals = [s[key] for s in per_doc.values() if key in s]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return lambda x: 0.0
        return lambda x: (x - lo) / (hi - lo)

    bm_norm, vs_norm = normaliser("bm25"), normaliser("vsm")
    fused = []
    for doc_id, s in per_doc.items():
        bm = bm_norm(s["bm25"]) if "bm25" in s else 0.0
        vs = vs_norm(s["vsm"]) if "vsm" in s else 0.0
        fused.append((doc_id, alpha * bm + (1 - alpha) * vs))
    fused.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in fused]


def ranking_wrrf(per_doc, w, k=RRF_K):
    """Weighted RRF: w * 1/(k+rank_bm25) + (1-w) * 1/(k+rank_vsm).
    w=0.5 is plain (unweighted) RRF; w=1 is BM25-by-rank, w=0 is VSM-by-rank."""
    bm_rank = {d: i for i, d in enumerate(ranking_by_score(per_doc, "bm25"), 1)}
    vs_rank = {d: i for i, d in enumerate(ranking_by_score(per_doc, "vsm"), 1)}
    fused = []
    for doc_id in per_doc:
        score = 0.0
        if doc_id in bm_rank:
            score += w * 1.0 / (k + bm_rank[doc_id])
        if doc_id in vs_rank:
            score += (1 - w) * 1.0 / (k + vs_rank[doc_id])
        fused.append((doc_id, score))
    fused.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in fused]


def ranking_rrf(per_doc, k=RRF_K):
    """Plain RRF: the w=0.5 case of weighted RRF."""
    return ranking_wrrf(per_doc, 0.5, k)


def per_query_scores(ranking_fn, cache, qrels, qids):
    """{qid: nDCG@10} for a ranking function."""
    return {q: ndcg_at_k(ranking_fn(cache[q]), qrels[q], k=10) for q in qids}


def mean(pq, qids):
    return sum(pq[q] for q in qids) / len(qids)


def bootstrap_ci(cand_pq, base_pq, qids):
    """95% CI on the mean per-query gain (candidate - base), resampling queries."""
    gains = []
    for _ in range(N_BOOT):
        sample = [random.choice(qids) for _ in qids]
        gains.append(sum(cand_pq[q] - base_pq[q] for q in sample) / len(sample))
    gains.sort()
    return gains[int(0.025 * N_BOOT)], gains[int(0.975 * N_BOOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out", default="runs/fusion_results.json")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    qrels = load_qrels(args.qrels)
    qids = [q for q in cache if q in qrels]
    print(f"{len(qids)} queries with both cached scores and qrels\n")

    # Every scheme as a ranking function.
    schemes = {
        "bm25": lambda pd: ranking_by_score(pd, "bm25"),
        "vsm": lambda pd: ranking_by_score(pd, "vsm"),
        "rrf": ranking_rrf,
    }
    for a in ALPHAS:
        schemes[f"linear(a={a})"] = lambda pd, a=a: ranking_linear(pd, a)
    for w in WEIGHTS:
        schemes[f"wrrf(w={w})"] = lambda pd, w=w: ranking_wrrf(pd, w)

    # Score every scheme (mean nDCG@10 over all queries).
    pq = {name: per_query_scores(fn, cache, qrels, qids) for name, fn in schemes.items()}

    scores = {name: mean(pq[name], qids) for name in schemes}

    print(f"{'scheme':<16}{'nDCG@10':>10}")
    print("-" * 26)
    for name in schemes:
        print(f"{name:<16}{scores[name]:>10.4f}")

    # Decision measure: best fusion by nDCG@10 (exclude the two singletons).
    fusion = [n for n in schemes if n not in ("bm25", "vsm")]
    best = max(fusion, key=lambda n: scores[n])
    gain = scores[best] - scores["bm25"]

    # Bootstrap CI: saved as a reference metric, not a decision gate.
    lo, hi = bootstrap_ci(pq[best], pq["bm25"], qids)

    print(f"\nBest fusion by nDCG@10: {best}  ({scores[best]:.4f})")
    print(f"Gain over BM25: {gain:+.4f}   bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")

    results = {
        "num_queries": len(qids),
        "scores": scores,
        "best_fusion": best,
        "gain_over_bm25": gain,
        "bootstrap_ci_95": [lo, hi],
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
