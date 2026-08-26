"""
submission/custom_scorer.py — optional combined/custom scorer.

Not required, but this is explicitly called out in the assignment
(Section 4.1) as "where separation in the leaderboard tends to happen":
any linear or non-linear combination of your Boolean/VSM and BM25
signals, additional features (e.g. proximity/bigram overlap), or your
own heuristic.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
from typing import List, Tuple

from submission.indexer import InvertedIndex
from submission import bm25
from submission import boolean_vsm

# Reciprocal Rank Fusion of BM25 and VSM. Chosen over a tuned linear
# combination because it is scale-free (fuses ranks, not the mismatched
# BM25 / cosine magnitudes) and parameter-free (k is a fixed convention),
# so there is nothing fitted on the 50 dev queries to overfit. On the dev
# set it scored nDCG@10 0.6405 vs BM25-alone 0.5900 (+0.0505). See
# scripts/fuse_scores.py and runs/fusion_results.json.
RRF_K = 60
BM25_K1 = 1.8
BM25_B = 0.6

# Only the top of each ranking matters to RRF: with k=60, a doc at rank
# 1000 contributes 1/1060 ~ 0.0009, far below anything that reaches the
# top 10. So we fuse each retriever's top-CAND candidates instead of all
# ~150K, cutting per-query latency with negligible effect on nDCG@10.
_CAND = 1000


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(). Just prepares the two underlying
    scorers against the same loaded index; RRF needs no state of its own."""
    bm25.build(index)
    boolean_vsm.build(index)


def _ranks(hits: List[Tuple[str, float]]) -> dict:
    """doc_id -> 1-based rank from a (doc_id, score) list, best first."""
    return {doc_id: i for i, (doc_id, _) in enumerate(hits, start=1)}


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, rrf_score) pairs for `query`, best first."""
    bm_ranks = _ranks(bm25.score(query, _CAND, k1=BM25_K1, b=BM25_B))
    vs_ranks = _ranks(boolean_vsm.vsm_score(query, _CAND))

    fused = {}
    for doc_id, r in bm_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + r)
    for doc_id, r in vs_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + r)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
