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

from submission.indexer import InvertedIndex, tokenize
from submission import bm25
from submission import boolean_vsm

# Weighted Reciprocal Rank Fusion of BM25 and VSM: chosen over a tuned
# linear combination because it is scale-free (fuses ranks, not the
# mismatched BM25 / cosine magnitudes). The single weight W_BM tilts the
# blend toward the stronger retriever (BM25); a dev sweep peaked at 0.6
# (nDCG@10 0.6453 vs plain-RRF 0.6405 vs BM25-alone 0.5900). See
# scripts/fuse_scores.py and runs/fusion_results.json.
RRF_K = 60
W_BM = 0.6  # weight on BM25's rank term; VSM gets (1 - W_BM)
BM25_K1 = 1.8
BM25_B = 0.6

# Only the top of each ranking matters to RRF: with k=60, a doc at rank
# 1000 contributes 1/1060 ~ 0.0009, far below anything that reaches the
# top 10. So we fuse each retriever's top-CAND candidates instead of all
# ~150K, cutting per-query latency with negligible effect on nDCG@10.
_CAND = 1000


# When a native (C++) index is loaded, scoring runs there; otherwise the
# pure-Python bm25 / boolean_vsm modules are used. Fusion (below) is the same
# cheap rank merge either way.
_NATIVE = None


def build(index: InvertedIndex) -> None:
    """Python-path setup: prepare the two pure-Python scorers against the
    loaded index. Called from retrieve.load_index() when the native
    extension isn't available."""
    bm25.build(index)
    boolean_vsm.build(index)


def build_native(index_dir: str) -> None:
    """Native-path setup: construct the C++ index directly from index_dir.
    No Python InvertedIndex is built — the C++ side owns the postings and
    does BM25/VSM scoring itself."""
    global _NATIVE
    from submission._index_cpp import NativeIndex

    _NATIVE = NativeIndex(index_dir)


def _ranks(hits: List[Tuple[str, float]]) -> dict:
    """doc_id -> 1-based rank from a (doc_id, score) list, best first."""
    return {doc_id: i for i, (doc_id, _) in enumerate(hits, start=1)}


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, rrf_score) pairs for `query`, best first.
    BM25/VSM come from the native index when present, else from the Python
    scorers; the weighted-RRF fusion is identical either way."""
    if _NATIVE is not None:
        tokens = tokenize(query)
        bm_hits = _NATIVE.bm25(tokens, BM25_K1, BM25_B, _CAND)
        vs_hits = _NATIVE.vsm(tokens, _CAND)
    else:
        bm_hits = bm25.score(query, _CAND, k1=BM25_K1, b=BM25_B)
        vs_hits = boolean_vsm.vsm_score(query, _CAND)

    bm_ranks = _ranks(bm_hits)
    vs_ranks = _ranks(vs_hits)
    fused = {}
    for doc_id, r in bm_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + W_BM / (RRF_K + r)
    for doc_id, r in vs_ranks.items():
        fused[doc_id] = fused.get(doc_id, 0.0) + (1 - W_BM) / (RRF_K + r)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
