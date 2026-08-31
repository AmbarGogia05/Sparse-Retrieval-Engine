"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces are provided:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""

import math
from collections import Counter, defaultdict
from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX = None
_DOC_NORM_SQUARED = {}


def build(index: InvertedIndex) -> None:
    """Prepare VSM state for the pure-Python fallback scorer.

    The native path does not call this function; custom_scorer.build() invokes
    it after loading the compact index when the native extension is absent.
    """
    global _INDEX, _DOC_NORM_SQUARED
    _INDEX = index
    _DOC_NORM_SQUARED = defaultdict(float)
    for postings in index.postings.values():
        inverse_document_frequency = math.log(index.N / len(postings))
        for doc_id, term_frequency in postings.items():
            _DOC_NORM_SQUARED[doc_id] += (
                term_frequency * inverse_document_frequency
            ) ** 2


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode}")
    query_tokens = tokenize(query)
    if mode == "and":
        if not query_tokens:
            return []
        for token in query_tokens:
            if token not in _INDEX.postings:
                return []  # no documents contain this term
        result_docs = set(_INDEX.postings[query_tokens[0]].keys())
        for token in query_tokens[1:]:
            result_docs.intersection_update(_INDEX.postings[token].keys())
        return list(result_docs)
    result_docs = set()
    for token in query_tokens:
        result_docs.update(_INDEX.postings.get(token, {}).keys())
    return list(result_docs)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    query_vector = {}
    for token, term_frequency in Counter(tokenize(query)).items():
        if token in _INDEX.postings:
            idf = math.log(_INDEX.N / _INDEX.document_frequency(token))
            query_vector[token] = term_frequency * idf

    scores = defaultdict(float)
    query_norm = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    for token, q_weight in query_vector.items():
        score_token = q_weight / query_norm if query_norm > 0 else 0
        postings = _INDEX.postings[token]
        idf = math.log(_INDEX.N / len(postings))
        for doc_id, term_frequency in postings.items():
            document_weight = term_frequency * idf
            scores[doc_id] += score_token * (
                document_weight / math.sqrt(_DOC_NORM_SQUARED[doc_id])
            )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
