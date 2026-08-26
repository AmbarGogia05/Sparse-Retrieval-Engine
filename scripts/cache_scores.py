"""
scripts/cache_scores.py — run BM25 and VSM once, cache their raw per-query,
per-candidate scores to disk so ensembling experiments are pure post-processing.

Retrieval over 171K docs is the slow part (~2 min build). Ensembling is just
arithmetic. So we do the expensive part once here and dump everything to a
JSON file; fuse_scores.py (and anything else) then reads that file and tries
fusion schemes in milliseconds without ever touching the corpus again.

Important: we ask each scorer for ALL candidates, not the top-10. If BM25
ranks a doc highly but VSM left it out of its top-10, we still need VSM's
score for that doc to fuse fairly. Both scorers only ever touch docs that
contain a query term, so "all candidates" is cheap and the two candidate
sets nearly coincide anyway.

Usage:
    python -m scripts.cache_scores \
        --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv \
        --out runs/score_cache.json
"""

import argparse
import json

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex
from submission import bm25
from submission import boolean_vsm

BIG_K = 10_000_000  # larger than any query's candidate count -> "give me all of them"


def read_queries(path):
    """Read queries from a TSV file (qid<TAB>text per line)."""
    queries = []
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries.append((qid, text))
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k1", type=float, default=1.8)
    ap.add_argument("--b", type=float, default=0.6)
    args = ap.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} docs")

    print("Building index...")
    index = InvertedIndex()
    index.build(corpus)

    # Wire up both scorers against the same in-memory index.
    bm25.build(index)
    boolean_vsm.build(index)

    queries = read_queries(args.queries)
    print(f"Scoring {len(queries)} queries with BM25 and VSM...")

    # cache[qid] = {doc_id: {"bm25": score, "vsm": score}}
    cache = {}
    for qid, text in queries:
        bm25_hits = bm25.score(text, BIG_K, k1=args.k1, b=args.b)
        vsm_hits = boolean_vsm.vsm_score(text, BIG_K)

        per_doc = {}
        for doc_id, s in bm25_hits:
            per_doc.setdefault(doc_id, {})["bm25"] = s
        for doc_id, s in vsm_hits:
            per_doc.setdefault(doc_id, {})["vsm"] = s

        cache[qid] = per_doc
        print(f"  {qid}: {len(bm25_hits)} bm25, {len(vsm_hits)} vsm, {len(per_doc)} union")

    meta = {"k1": args.k1, "b": args.b, "num_queries": len(queries)}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "scores": cache}, f)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
