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

# W_BM is deliberately off its dev optimum (0.6). Off-domain corpora want the
# VSM arm nearly off (best W_BM: nfcorpus 1.0, fiqa 1.0, antique 0.9 vs COVID
# 0.6), and 0.6 would cost -0.022 on antique. 0.7 gives up 0.007 on dev, inside
# the ~0.01 noise floor of 50 queries. Graded topics are on a held-out, non-
# COVID collection (spec Section 6). Full table: runs/wbm_crosscorpus.md
W_BM = float(os.environ.get("SRE_W_BM", 0.7))  # VSM gets 1 - W_BM

# RM3 pseudo-relevance feedback, native path only (needs the forward index):
# round-1 BM25 -> relevance model over the top-R docs -> interpolate with the
# original query at weight RM3_LAMBDA -> weighted round-2 BM25. Plateau pick
# from scripts/tune_rm3.py, not the grid argmax.
RM3_R = int(os.environ.get("SRE_RM3_R", 10))
RM3_M = int(os.environ.get("SRE_RM3_M", 15))
RM3_LAMBDA = float(os.environ.get("SRE_RM3_LAMBDA", 0.6))

# Feedback-document weights are a softmax of the round-1 BM25 score over the
# top-R docs (normalised by the top score, so the weighting is scale-free
# across corpora) rather than score/sum, which lets one anomalous top document
# dominate the relevance model. RM3_NOVEL keeps the M expansion slots for terms
# the query doesn't already contain. The two are superadditive: +0.002 each,
# +0.009 together, pooled over 1221 queries on four corpora. The temperature is
# the interior of the [0.10, 0.20] band that is non-negative on every corpus,
# not any single corpus's argmax. See runs/rm3_feedback_weighting.md
RM3_FB_TEMP = float(os.environ.get("SRE_RM3_FB_TEMP", 0.15))
RM3_NOVEL = os.environ.get("SRE_RM3_NOVEL", "1") != "0"

# At RRF_K=60 a doc at rank 1000 contributes 1/1060, far below anything that
# reaches the top 10 — so fusing top-1000 per arm costs no measurable nDCG@10.
_CAND = 1000


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
        if USE_RM3 and getattr(_NATIVE, "has_forward", False):
            bm_hits = _NATIVE.rm3(tokens, RM3_R, RM3_M, RM3_LAMBDA,
                                  BM25_K1, BM25_B, _CAND,
                                  RM3_FB_TEMP, RM3_NOVEL)
        else:
            bm_hits = _NATIVE.bm25(tokens, BM25_K1, BM25_B, _CAND)
        vs_hits = _NATIVE.vsm(tokens, _CAND)
    else:
        bm_hits = bm25.score(query, _CAND, k1=BM25_K1, b=BM25_B)
        vs_hits = boolean_vsm.vsm_score(query, _CAND)

    fused = {}
    for hits, w in ((bm_hits, W_BM), (vs_hits, 1 - W_BM)):
        for doc_id, r in _ranks(hits).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (RRF_K + r)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
