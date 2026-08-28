"""THE REQUIRED COMPETITION ENTRYPOINT — do not rename these three functions
or change their signatures (assignment Section 5).

    build_index(corpus_path, index_dir)  build and persist; timed, and the
                                         on-disk size of index_dir is graded
    load_index(index_dir)                rebuild query-time state from disk only
    retrieve(query, k)                   -> up to k (doc_id, score), best first

build_index() and load_index()/retrieve() run in SEPARATE processes: nothing
survives between them except what is written to index_dir.
"""

import json
import os
from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex
from submission import custom_scorer

# load_index() populates this; retrieve() reads it.
_INDEX = None

# SPIMI: flush the in-memory postings block once it holds this many postings.
# ~40M postings ≈ 1.5 GB accumulator — well within an 8 GB budget alongside
# the corpus text and interpreter, with headroom for the merge phase. Small
# corpora (e.g. 171K docs ≈ 17M postings) stay a single in-memory block.
_SPIMI_FLUSH_POSTINGS = 40_000_000

# Truecase gate threshold; must match InvertedIndex.save_v2's 0.5.
_CASE_DF_RATIO = 0.5


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build the index and persist it to `index_dir`. Timed."""
    corpus = load_corpus(corpus_path)

    # C++ SPIMI: flushes sorted blocks past a threshold and merges them into
    # save_v2's on-disk format, so peak memory stays bounded on large corpora.
    # Tokenisation stays in Python (memoised nltk).
    try:
        from submission._spimi_cpp import SpimiBuilder
        from submission.indexer import tokenize_doc, _stem

        builder = SpimiBuilder(index_dir, _SPIMI_FLUSH_POSTINGS)
        for doc_id, text in corpus:
            tokens, doc_len = tokenize_doc(text)
            builder.add_document(doc_id, tokens, doc_len)
        canonical = {t: _stem(t.lower()) for t in builder.case_terms()}
        builder.finalize(canonical, _CASE_DF_RATIO)
    except ImportError:
        index = InvertedIndex()
        index.build(corpus)
        index.save_v2(index_dir)  # JSON save() kept in indexer.py for reference


def load_index(index_dir: str) -> None:
    """Reconstruct query-time state, reading only from `index_dir`. Timed."""
    global _INDEX

    # Prefer the native (C++) index: decodes postings and scores in C++,
    # avoiding both the Python dict rebuild at load and the per-query Python
    # loops. Falls back to InvertedIndex + bm25/boolean_vsm if unbuilt.
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

    return custom_scorer.score(query, k)
