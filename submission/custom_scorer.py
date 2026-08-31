"""Combined scorer: weighted RRF fusion of an RM3/BM25 arm and a VSM arm.
This is the shipped competition entry (assignment Section 4.1)."""
import os
from typing import List, Tuple

from submission import bm25
from submission import boolean_vsm
from submission.indexer import InvertedIndex

USE_RM3 = os.environ.get("SRE_RM3", "1") != "0"

# RRF fuses ranks, not scores, so the mismatched BM25/cosine magnitudes never
# have to be normalised. See scripts/fuse_scores.py, runs/fusion_results.json.
RRF_K = 60
BM25_K1 = 1.8
BM25_B = 0.6

# Shipped fusion weight for the BM25 arm; override for local experiments.
W_BM = float(os.environ.get("SRE_W_BM", 0.85))  # VSM gets 1 - W_BM

# RM3 pseudo-relevance feedback, native path only (needs the reconstructed
# forward view): round-1 BM25 -> relevance model over the top-R docs ->
# interpolation with the original query -> weighted round-2 BM25.
RM3_R = int(os.environ.get("SRE_RM3_R", 15))
RM3_M = int(os.environ.get("SRE_RM3_M", 30))
RM3_LAMBDA = float(os.environ.get("SRE_RM3_LAMBDA", 0.6))
RM3_K1_ROUND2 = float(os.environ.get("SRE_RM3_K1_ROUND2", 1.2))
RM3_B_ROUND2 = float(os.environ.get("SRE_RM3_B_ROUND2", 0.4))

# Feedback-document weights are a softmax of the round-1 BM25 score over the
# top-R docs, normalised by the top score. RM3_NOVEL reserves the M expansion
# slots for terms not already present in the query. Both settings are
# configurable for experiments.
RM3_FB_TEMP = float(os.environ.get("SRE_RM3_FB_TEMP", 0.10))
RM3_NOVEL = os.environ.get("SRE_RM3_NOVEL", "1") != "0"

# At RRF_K=60 a document ranked beyond 500 in both arms cannot surface in the
# fused top ten.
_CAND = 500
# Small IDF-weighted distinct-query-term coverage rerank on only the fused top
# ten.
_COVERAGE_RERANK_ALPHA = 0.30
_COVERAGE_RERANK_DEPTH = 10
# Conservative early-document presence tie-breaker. A sparse side stream records
# which ordinary postings occur in the first 12 raw words; the signal is used
# only on the existing fused top ten.
_EARLY_RERANK_ALPHA = 0.05


# Native (C++) index when available, else the pure-Python scorers.
_NATIVE = None


def build(index: InvertedIndex) -> None:
    """Python-path setup, used when the native extension isn't available."""
    bm25.build(index)
    boolean_vsm.build(index)


def build_native(index_dir: str) -> None:
    """Native-path setup: the C++ side owns the postings and does the scoring."""
    global _NATIVE
    from submission._index_cpp import NativeIndex
    from submission.indexer import _STEMMED_STOPWORDS

    _NATIVE = NativeIndex(index_dir, list(_STEMMED_STOPWORDS))


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, final_score) pairs for `query`, best first."""
    if _NATIVE is not None:
        # Exactly one pybind call in and one result conversion out. NativeIndex
        # tokenises the raw query, runs both scoring arms, computes the two
        # rerank signals, fuses, reranks, and returns only the final top-k.
        return _NATIVE.retrieve(
            query, k, USE_RM3, RM3_R, RM3_M, RM3_LAMBDA,
            BM25_K1, BM25_B, _CAND, RM3_FB_TEMP, RM3_NOVEL,
            RM3_K1_ROUND2, RM3_B_ROUND2, RRF_K, W_BM,
            _COVERAGE_RERANK_ALPHA, _COVERAGE_RERANK_DEPTH,
            _EARLY_RERANK_ALPHA,
        )

    bm_hits = bm25.score(query, _CAND, k1=BM25_K1, b=BM25_B)
    vs_hits = boolean_vsm.vsm_score(query, _CAND)
    fused = {}
    for hits, weight in ((bm_hits, W_BM), (vs_hits, 1 - W_BM)):
        for rank, (doc_id, _) in enumerate(hits, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (RRF_K + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
