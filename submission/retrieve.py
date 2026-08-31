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

# Hard cap, not os.cpu_count(): the grading machine has 4 cores, and local
# build-time numbers should reflect that rather than this dev machine's
# core count.
_BUILD_WORKERS = 4

# Truecase gate threshold; must match InvertedIndex.save_v2's 0.5.
_CASE_DF_RATIO = 0.5


def _doc_line_offsets(corpus_path: str) -> List[int]:
    """Byte offset of the start of each non-blank line in corpus_path, in
    file order. Doc ordinal i (as load_corpus/the serial build assigns it)
    corresponds to offsets[i] — load_corpus also skips blank lines, so a
    blank line must not consume an ordinal here either."""
    offsets = []
    with open(corpus_path, "rb") as f:
        pos = f.tell()
        for raw in f:
            if raw.strip():
                offsets.append(pos)
            pos = f.tell()
    return offsets


def _spimi_worker(args):
    """Runs in a worker process: tokenises and SPIMI-indexes one contiguous,
    globally-ordinalled slice of the corpus into its own scratch subdir.
    Returns only the paths and counts needed by the parent merge."""
    (
        corpus_path,
        start_byte,
        n_docs_to_read,
        start_ord,
        worker_dir,
        flush_threshold,
    ) = args
    from submission._spimi_cpp import SpimiBuilder
    from submission import indexer

    os.makedirs(worker_dir, exist_ok=True)
    builder = SpimiBuilder(worker_dir, flush_threshold, start_ord)
    # Fused path: the builder tokenises+stems each doc in C++ (no Python token
    # list). Needs its own stopword set. Falls back to the Python tokenizer when
    # the C++ tokenizer isn't active (SRE_SNOWBALL / experimental knobs).
    use_cpp = indexer._USE_CPP_TOKENIZE
    if use_cpp:
        builder.set_stopword_stems(list(indexer._STEMMED_STOPWORDS))
    with open(corpus_path, "rb") as f:
        f.seek(start_byte)
        count = 0
        while count < n_docs_to_read:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if use_cpp:
                builder.add_document_from_text(obj["doc_id"], obj["text"])
            else:
                tokens, doc_len = indexer.tokenize_doc(obj["text"])
                builder.add_document(obj["doc_id"], tokens, doc_len)
            count += 1
    builder.close_worker()

    return {
        "block_files": builder.block_files(),
        "docs_txt": builder.docs_path(),
        "doclen_bin": builder.doclen_path(),
        "n_docs": builder.n_docs_count(),
        "total_len": builder.total_len_count(),
        "case_terms": builder.case_terms(),
    }


def _build_index_parallel(corpus_path: str, index_dir: str) -> bool:
    """Attempt the 4-worker-process SPIMI build. Returns True on success,
    False if the parallel path can't be used here (e.g. a sandboxed runner
    that can't start a process pool) so the caller can fall back to the
    serial path. Must produce byte-identical output to the serial build —
    see the extensive comments in _spimi_cpp.cpp's finalize_parallel."""
    import multiprocessing as mp
    from submission._spimi_cpp import finalize_parallel
    from submission.indexer import _stem

    offsets = _doc_line_offsets(corpus_path)
    n = len(offsets)
    if n == 0:
        finalize_parallel([], index_dir, {}, _CASE_DF_RATIO)
        return True

    n_workers = min(_BUILD_WORKERS, n)
    base, rem = divmod(n, n_workers)
    worker_flush = max(1, _SPIMI_FLUSH_POSTINGS // _BUILD_WORKERS)

    tasks = []
    worker_dirs = []
    start_ord = 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        worker_dir = os.path.join(index_dir, f"_w{i}")
        worker_dirs.append(worker_dir)
        tasks.append(
            (
                corpus_path,
                offsets[start_ord],
                size,
                start_ord,
                worker_dir,
                worker_flush,
            )
        )
        start_ord += size

    ctx = mp.get_context()
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_spimi_worker, tasks)

    all_case_terms = set()
    for r in results:
        all_case_terms.update(r["case_terms"])
    canonical = {t: _stem(t.lower()) for t in all_case_terms}

    finalize_parallel(results, index_dir, canonical, _CASE_DF_RATIO)

    for worker_dir in worker_dirs:
        try:
            os.rmdir(worker_dir)
        except OSError:
            pass
    return True


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build the index and persist it to `index_dir`. Timed."""
    # Parallel path: 4 worker processes each tokenise and SPIMI-build their
    # own contiguous, globally-ordinalled slice of the corpus; the parent
    # only does the final (single-threaded) k-way merge and write. This is
    # purely a build-time optimisation — output must be byte-identical to
    # the serial path below (see finalize_parallel in _spimi_cpp.cpp for how
    # that's achieved, and _build_index_parallel for the corpus splitting).
    # Falls back to the serial path if a process pool can't be started here
    # (e.g. a sandboxed/restricted CI runner) or the C++ extension isn't
    # built at all.
    try:
        if _build_index_parallel(corpus_path, index_dir):
            return
    except ImportError:
        pass
    except Exception:
        # Any other failure to stand up the process pool (sandboxed runner,
        # OS refusing fork/spawn, etc.) — CI conformance matters more than
        # the speedup, so fall through to the proven serial path below.
        pass

    corpus = load_corpus(corpus_path)

    # C++ SPIMI: flushes sorted blocks past a threshold and merges them into
    # compact v2 on-disk format, so peak memory stays bounded on large corpora.
    # Tokenisation stays in Python (memoised nltk).
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


def load_index(index_dir: str) -> None:
    """Reconstruct query-time state, reading only from `index_dir`. Timed."""
    global _INDEX

    # Prefer the native (C++) index: decodes postings and scores in C++,
    # avoiding both the Python dict rebuild at load and the per-query Python
    # loops. Falls back to InvertedIndex + bm25/boolean_vsm if unbuilt.
    try:
        import submission._index_cpp  # noqa: F401  (presence check)
        from submission import indexer

        with open(f"{index_dir}/meta.json") as f:
            meta = json.load(f)
        indexer.set_case_terms(indexer._load_case_terms(index_dir, meta))
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
