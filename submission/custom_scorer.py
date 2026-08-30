"""Combined scorer: weighted RRF fusion of an RM3-expanded BM25 arm and a VSM
arm. This is the shipped competition entry (assignment Section 4.1)."""
import os
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
from submission import bm25
from submission import boolean_vsm

USE_RM3 = os.environ.get("SRE_RM3", "1") != "0"

# RRF fuses ranks, not scores, so the mismatched BM25/cosine magnitudes never
# have to be normalised. See scripts/fuse_scores.py, runs/fusion_results.json.
RRF_K = 60
BM25_K1 = 1.8
BM25_B = 0.6

# The candidate-era cross-corpus sweep selected .85 using the non-COVID probes:
# it improves NFCorpus, FiQA, and Antique over .70, while changing the public
# COVID proxy only slightly. This is chosen from generalization probes rather
# than a COVID-only argmax.
W_BM = float(os.environ.get("SRE_W_BM", 0.85))  # VSM gets 1 - W_BM

# RM3 pseudo-relevance feedback, native path only (needs the forward index):
# round-1 BM25 -> relevance model over the top-R docs -> interpolate with the
# original query at weight RM3_LAMBDA -> weighted round-2 BM25. These are the
# candidate-era generalization settings.
RM3_R = int(os.environ.get("SRE_RM3_R", 15))
RM3_M = int(os.environ.get("SRE_RM3_M", 30))
RM3_LAMBDA = float(os.environ.get("SRE_RM3_LAMBDA", 0.6))
RM3_K1_ROUND2 = float(os.environ.get("SRE_RM3_K1_ROUND2", 1.2))
RM3_B_ROUND2 = float(os.environ.get("SRE_RM3_B_ROUND2", 0.4))

# Feedback-document weights are a softmax of the round-1 BM25 score over the
# top-R docs (normalised by the top score, so the weighting is scale-free
# across corpora) rather than score/sum, which lets one anomalous top document
# dominate the relevance model. RM3_NOVEL keeps the M expansion slots for terms
# the query doesn't already contain. The two are superadditive: +0.002 each,
# +0.009 together, pooled over 1221 queries on four corpora. The temperature is
# the interior of the [0.10, 0.20] band that is non-negative on every corpus,
# not any single corpus's argmax. See runs/rm3_feedback_weighting.md
RM3_FB_TEMP = float(os.environ.get("SRE_RM3_FB_TEMP", 0.10))
RM3_NOVEL = os.environ.get("SRE_RM3_NOVEL", "1") != "0"

# At RRF_K=60 a doc at rank 500 contributes 1/560, far below anything that
# reaches the fused top 10 — a doc ranked >500 in BOTH arms cannot surface. A
# A document ranked >500 in both arms cannot surface in the fused top ten.
_CAND = 500
# Small IDF-weighted distinct-query-term coverage rerank on only the fused top
# ten. The alpha/depth pair was positive on all four probe corpora.
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

    _NATIVE = NativeIndex(index_dir)


def _ranks(hits: List[Tuple[str, float]]) -> dict:
    """doc_id -> 1-based rank from a (doc_id, score) list, best first."""
    return {doc_id: i for i, (doc_id, _) in enumerate(hits, start=1)}


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, rrf_score) pairs for `query`, best first."""
    if _NATIVE is not None:
        tokens = tokenize(query)
        # VSM always scores the ORIGINAL query — the fusion pays for arm
        # disagreement, so expanding both arms would erode the gain.
        # arms() runs the RM3/BM25 and VSM arms on two C++ threads with the GIL
        # released, so the cheaper VSM arm hides under the RM3 arm's latency.
        # The rm3-vs-bm25 choice (USE_RM3 && has_forward) is made inside arms().
        bm_hits, vs_hits = _NATIVE.arms(
            tokens, USE_RM3, RM3_R, RM3_M, RM3_LAMBDA,
            BM25_K1, BM25_B, _CAND, RM3_FB_TEMP, RM3_NOVEL,
            RM3_K1_ROUND2, RM3_B_ROUND2)
        coverage_hits = _NATIVE.coverage(tokens, _CAND, True)
        early_hits = _NATIVE.early_match(tokens, _CAND)
    else:
        bm_hits = bm25.score(query, _CAND, k1=BM25_K1, b=BM25_B)
        vs_hits = boolean_vsm.vsm_score(query, _CAND)

    fused = {}
    for hits, w in ((bm_hits, W_BM), (vs_hits, 1 - W_BM)):
        for doc_id, r in _ranks(hits).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (RRF_K + r)

    ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    if _NATIVE is not None and _COVERAGE_RERANK_ALPHA > 0.0:
        coverage = dict(coverage_hits)
        early = dict(early_hits)
        depth = min(_COVERAGE_RERANK_DEPTH, len(ranked))
        head = ranked[:depth]
        fmax = head[0][1] if head else 0.0
        cmax = max((coverage.get(doc_id, 0.0) for doc_id, _ in head), default=0.0)
        if fmax > 0.0 and cmax > 0.0:
            head = [
                (
                    doc_id,
                    fused_score / fmax
                    + _COVERAGE_RERANK_ALPHA
                    * coverage.get(doc_id, 0.0) / cmax,
                )
                for doc_id, fused_score in head
            ]
            head.sort(key=lambda x: (-x[1], x[0]))
            ranked = head + ranked[depth:]
        if _EARLY_RERANK_ALPHA > 0.0:
            qden = max(1, len(set(tokens)))
            depth = min(_COVERAGE_RERANK_DEPTH, len(ranked))
            head = [
                (
                    doc_id,
                    fused_score + _EARLY_RERANK_ALPHA
                    * early.get(doc_id, 0.0) / qden,
                )
                for doc_id, fused_score in ranked[:depth]
            ]
            head.sort(key=lambda x: (-x[1], x[0]))
            ranked = head + ranked[depth:]
    return ranked[:k]
