"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

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

from typing import List, Tuple

from submission.indexer import InvertedIndex, tokenize
import math


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything VSM-specific (e.g. document vector
    norms) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX
    _INDEX = index
    global doc_norm_cache
    doc_norm_cache = {}
    for token in _INDEX.postings:
        idf_token = math.log(_INDEX.N / _INDEX.document_frequency(token))
        for doc_id, tf_token_doc in _INDEX.postings[token].items():
            if doc_id not in doc_norm_cache:
                doc_norm_cache[doc_id] = 0.0
            doc_norm_cache[doc_id] += (tf_token_doc * idf_token) ** 2


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode}")
    query_tokens = tokenize(query)
    if mode == "and":
        for token in query_tokens:
            if token not in _INDEX.postings:
                return []  # no documents contain this term
        # Start with the set of documents containing the first token
        result_docs = set(_INDEX.postings[query_tokens[0]].keys())
        for token in query_tokens[1:]:
            result_docs.intersection_update(_INDEX.postings[token].keys())
        return list(result_docs)
    else:  # mode == "or"
        result_docs = set()
        for token in query_tokens:
            if token in _INDEX.postings:
                result_docs.update(_INDEX.postings[token].keys())
        return list(result_docs)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    query_tokens = tokenize(query)
    query_vector = {}
    for token in query_tokens:
        if token in _INDEX.postings:
            tf = query_tokens.count(token)
            idf = math.log(_INDEX.N / _INDEX.document_frequency(token))
            query_vector[token] = tf * idf

    scores = {}
    query_norm = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    for token, q_weight in query_vector.items():
        score_token = q_weight / query_norm if query_norm > 0 else 0
        for doc_id in _INDEX.postings.get(token, {}):
            tf = _INDEX.postings[token][doc_id]
            idf = math.log(_INDEX.N / _INDEX.document_frequency(token))
            d_weight = tf * idf
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += score_token * (
                d_weight / math.sqrt(doc_norm_cache[doc_id])
            )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
