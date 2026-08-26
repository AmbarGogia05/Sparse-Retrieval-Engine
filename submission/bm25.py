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
from typing import List, Tuple
from submission.indexer import InvertedIndex, tokenize


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything BM25-specific (e.g. cached IDF values
    per term) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX
    _INDEX = index
    global idf_cache
    idf_cache = {}
    for token in _INDEX.postings:
        idf_cache[token] = math.log(
            (_INDEX.N - _INDEX.document_frequency(token) + 0.5)
            / (_INDEX.document_frequency(token) + 0.5)
            + 1
        )


def score(
    query: str, k: int, k1: float = 1.2, b: float = 0.75
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    query_tokens = tokenize(query)
    scores = {}
    for token in query_tokens:
        if token not in _INDEX.postings:
            continue
        idf_token = idf_cache[token]
        for doc_id in _INDEX.postings[token]:
            tf_token_doc = _INDEX.postings[token][doc_id]
            doc_len = _INDEX.doc_len[doc_id]
            avg_doc_len = _INDEX.avg_doc_len
            score_token_doc = (
                idf_token
                * (tf_token_doc * (k1 + 1))
                / (tf_token_doc + k1 * (1 - b + b * doc_len / avg_doc_len))
            )
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += score_token_doc
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
