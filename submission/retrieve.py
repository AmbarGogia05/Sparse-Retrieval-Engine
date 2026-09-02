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
import shutil
import tempfile
import zlib
from typing import List, Tuple

from submission import custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

# load_index() populates this; retrieve() reads it.
_INDEX = None

# SPIMI: flush the in-memory postings block once it reaches this threshold.
# The parallel build divides the threshold across its workers, bounding peak
# memory while preserving the same final compact v2 bytes as the serial path.
_SPIMI_FLUSH_POSTINGS = 40_000_000

# Hard cap: the grading machine has 4 cores, and local build-time numbers
# should reflect that rather than the development machine's core count.
_BUILD_WORKERS = 4

# Truecase gate threshold; must match InvertedIndex.save_v2's 0.5.
_CASE_DF_RATIO = 0.5

# These sidecars still contain enough byte-level redundancy for fast DEFLATE
# to save useful space. postings.bin is deliberately excluded: after the
# adaptive Rice codec it compresses only slightly, while dominating compression
# CPU. The loader expands sidecars into a temporary directory before handing
# them to the unchanged native/Python readers.
_COMPRESSIBLE_SIDECARS = (
    "terms.txt",
    "case_terms.bin",
    "doclen.bin",
    "early.bin",
    "docids.bin",
    "docs.txt",
    "tf1.bin",
)
def _compress_index_sidecars(index_dir: str) -> None:
    """Replace compressible sidecars with level-1 zlib streams when smaller."""
    for name in _COMPRESSIBLE_SIDECARS:
        path = os.path.join(index_dir, name)
        compressed_path = path + ".z"
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            raw = f.read()
        compressed = zlib.compress(raw, level=1)
        if len(compressed) >= len(raw):
            try:
                os.remove(compressed_path)
            except FileNotFoundError:
                pass
            continue

        temporary_path = compressed_path + ".tmp"
        with open(temporary_path, "wb") as f:
            f.write(compressed)
        os.replace(temporary_path, compressed_path)
        os.remove(path)


def _materialize_index_sidecars(index_dir: str):
    """Return a reader-compatible directory and an optional cleanup path."""
    compressed_names = [
        name for name in _COMPRESSIBLE_SIDECARS
        if os.path.isfile(os.path.join(index_dir, name + ".z"))
    ]
    if not compressed_names:
        return index_dir, None

    materialized = tempfile.mkdtemp(prefix="sre_index_load_")
    compressed_set = set(compressed_names)
    try:
        for entry in os.listdir(index_dir):
            source = os.path.join(index_dir, entry)
            if not os.path.isfile(source):
                continue
            if entry.endswith(".z") and entry[:-2] in compressed_set:
                target = os.path.join(materialized, entry[:-2])
                with open(source, "rb") as f:
                    data = zlib.decompress(f.read())
                with open(target, "wb") as f:
                    f.write(data)
            elif entry not in compressed_set:
                os.symlink(os.path.abspath(source),
                           os.path.join(materialized, entry))
    except Exception:
        shutil.rmtree(materialized, ignore_errors=True)
        raise
    return materialized, materialized


def _build_index_parallel(corpus_path: str, index_dir: str) -> bool:
    """Read, parse, tokenise, index, and merge entirely in C++.

    This is one pybind call: Python sends paths/configuration once, and no
    per-document strings, token lists, worker metadata, or intermediate index
    structures cross back through the binding.
    """
    from submission._spimi_cpp import build_index_from_jsonl
    from submission import indexer

    # Preserve the experimental pure-Python tokenizer switches.
    if not indexer._USE_CPP_TOKENIZE:
        return False

    worker_flush = max(1, _SPIMI_FLUSH_POSTINGS // _BUILD_WORKERS)
    build_index_from_jsonl(
        corpus_path,
        index_dir,
        list(indexer._STEMMED_STOPWORDS),
        worker_flush,
        _BUILD_WORKERS,
        _CASE_DF_RATIO,
    )
    return True


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build the index and persist it to `index_dir`. Timed."""
    # One native call reads and parses JSONL, tokenises, builds worker-local
    # SPIMI blocks, and performs the final merge. The corpus and intermediate
    # data never materialise as Python objects.
    try:
        if _build_index_parallel(corpus_path, index_dir):
            _compress_index_sidecars(index_dir)
            return
    except ImportError:
        pass

    corpus = load_corpus(corpus_path)

    # C++ SPIMI: flushes sorted blocks past a threshold and merges them into
    # compact v2 on-disk format, so peak memory stays bounded on large corpora.
    # This is retained as a compatibility fallback for an older extension that
    # lacks build_index_from_jsonl().
    try:
        from submission._spimi_cpp import SpimiBuilder
        from submission import indexer

        builder = SpimiBuilder(index_dir, _SPIMI_FLUSH_POSTINGS)
        use_cpp = indexer._USE_CPP_TOKENIZE
        if use_cpp:
            builder.set_stopword_stems(list(indexer._STEMMED_STOPWORDS))
        for doc_id, text in corpus:
            if use_cpp:
                builder.add_document_from_text(doc_id, text)
            else:
                tokens, doc_len = indexer.tokenize_doc(text)
                builder.add_document(doc_id, tokens, doc_len)
        canonical = {t: indexer._stem(t.lower()) for t in builder.case_terms()}
        builder.finalize(canonical, _CASE_DF_RATIO)
    except ImportError:
        index = InvertedIndex()
        index.build(corpus)
        index.save_v2(index_dir)
    _compress_index_sidecars(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct query-time state, reading only from `index_dir`. Timed."""
    global _INDEX

    reader_dir, cleanup_dir = _materialize_index_sidecars(index_dir)
    try:
        # Prefer the native (C++) index: decodes postings and scores in C++,
        # avoiding both the Python dict rebuild at load and the per-query Python
        # loops. Falls back to InvertedIndex + bm25/boolean_vsm if unbuilt.
        try:
            import submission._index_cpp  # noqa: F401  (presence check)
            from submission import indexer

            with open(f"{reader_dir}/meta.json") as f:
                meta = json.load(f)
            indexer.set_case_terms(
                indexer._load_case_terms(reader_dir, meta))
            custom_scorer.build_native(reader_dir)
            _INDEX = "native"  # non-None sentinel; scoring lives in custom_scorer
        except ImportError:
            _INDEX = InvertedIndex.load_v2(reader_dir)
            custom_scorer.build(_INDEX)
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


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
