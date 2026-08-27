"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in the order build_index() saw them, ignoring the query
entirely — wired up below. It actually persists to disk and reloads
correctly, so it exercises the full build -> disk -> fresh process -> load
-> query path end-to-end from your very first commit. Its scores will be
close to zero; replace the logic, but keep the same
persist-in-build / reconstruct-in-load shape.
"""

import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex
from submission import custom_scorer

# TODO(you): once implemented, import and use your real scorers, e.g.:
# from submission import bm25, boolean_vsm, custom_scorer
# from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this; retrieve() reads it.
# build_index() runs in a SEPARATE process and cannot rely on this state
# surviving into load_index()/retrieve() — anything needed at query time
# must be written to index_dir in build_index() and read back in
# load_index().
# ---------------------------------------------------------------------------
_INDEX = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build whatever index structures you need, and
    write everything retrieve() will need into `index_dir`.

    Runs once, in its own process, before load_index() ever runs. Heavy
    one-time work — tokenising the whole corpus, building postings lists,
    computing collection statistics — belongs here, not in retrieve(), so
    it doesn't get charged against your per-query latency. Whatever you
    don't write to `index_dir` here does not exist as far as load_index()
    is concerned.
    """
    corpus = load_corpus(corpus_path)

    # TODO(you): build your real inverted index / term statistics here, e.g.:
    #
    #   from submission.indexer import InvertedIndex
    index = InvertedIndex()
    index.build(corpus)
    # Compressed persistence (int doc-ids + gap + VByte). The JSON baseline
    # index.save(index_dir) is kept in indexer.py for reference/testing.
    index.save_v2(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct everything retrieve() needs, reading only from
    `index_dir`. Runs once, in a fresh process, before any retrieve()
    calls — there is no leftover state from build_index() to rely on.
    """
    global _INDEX

    # TODO(you): load your real index here, e.g.:
    #
    #   from submission.indexer import InvertedIndex
    #   index = InvertedIndex.load(index_dir)
    #   bm25.build(index)
    #   boolean_vsm.build(index)
    #
    # and store it in a module-level variable so retrieve() can use it.
    # Prefer the native (C++) index: it decodes the postings and does BM25/VSM
    # scoring in C++, avoiding both the Python dict reconstruction at load and
    # the Python scoring loops per query. Falls back to the pure-Python
    # InvertedIndex + bm25/boolean_vsm path if the extension isn't built.
    try:
        import submission._index_cpp  # noqa: F401  (presence check)
        import json
        from submission.indexer import set_case_terms

        with open(f"{index_dir}/meta.json") as f:
            set_case_terms(set(json.load(f).get("case_terms", [])))
        custom_scorer.build_native(index_dir)
        _INDEX = "native"  # non-None sentinel; scoring lives in custom_scorer
    except ImportError:
        _INDEX = InvertedIndex.load_v2(index_dir)
        custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    # RRF fusion of BM25 (k1=1.8, b=0.6) and VSM. See custom_scorer.py and
    # runs/fusion_results.json: dev nDCG@10 0.6405 vs BM25-alone 0.5900.
    return custom_scorer.score(query, k)
