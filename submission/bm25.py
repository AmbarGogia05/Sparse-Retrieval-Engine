"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""

import math
from collections import defaultdict
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX = None
_IDF = {}


def build(index: InvertedIndex) -> None:
    """Prepare BM25 state for the pure-Python fallback scorer.

    The native path does not call this function; custom_scorer.build() invokes
    it after loading the compact index when the native extension is absent.
    """
    global _INDEX, _IDF
    _INDEX = index
    _IDF = {}
    for term, postings in index.postings.items():
        document_frequency = len(postings)
        _IDF[term] = math.log(
            (index.N - document_frequency + 0.5)
            / (document_frequency + 0.5)
            + 1
        )


def score(
    query: str, k: int, k1: float = 1.2, b: float = 0.75
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    scores = defaultdict(float)
    for term in tokenize(query):
        postings = _INDEX.postings.get(term)
        if postings is None:
            continue
        for doc_id, term_frequency in postings.items():
            length_ratio = _INDEX.doc_len[doc_id] / _INDEX.avg_doc_len
            scores[doc_id] += (
                _IDF[term]
                * term_frequency
                * (k1 + 1)
                / (
                    term_frequency
                    + k1 * (1 - b + b * length_ratio)
                )
            )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
