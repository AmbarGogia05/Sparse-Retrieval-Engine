"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""

import re
from typing import Dict, List, Tuple
import json
from nltk.stem.snowball import SnowballStemmer

_STEMMER = SnowballStemmer("english")

# Snowball stemming dominates build time (~95%), and the tokenizer re-stems
# every token occurrence — so common words like "the"/"covid" get stemmed
# millions of times. Memoise on the raw word: identical output, ~5x faster
# build. The cache is per-process and unbounded (vocab is finite, ~10^5).
_STEM_CACHE = {}


def _stem(word: str) -> str:
    s = _STEM_CACHE.get(word)
    if s is None:
        s = _STEMMER.stem(word)
        _STEM_CACHE[word] = s
    return s


_HYPHEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HYPHEN_RE_CASE = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+")
_TOKEN_RE_CASE = re.compile(r"[a-zA-Z0-9]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+|[a-zA-Z0-9]+")


def _has_internal_upper(s: str) -> bool:
    return any(c.isupper() for c in s[1:])


def tokenize_v0(text: str) -> List[str]:
    lower = text.lower()
    tokens = []
    for m in _HYPHEN_RE.finditer(lower):
        tokens.append(_STEMMER.stem(m.group()))
    for m in _TOKEN_RE.findall(lower):
        tokens.append(_STEMMER.stem(m))
    return tokens


_CASE_TERMS: set = None


def _is_case_candidate(word: str) -> bool:
    return len(word) >= 2 and _has_internal_upper(word)


def tokenize_v1(text: str) -> Tuple[List[str], int]:
    tokens = []
    primary_count = 0
    for m in _WORD_RE.finditer(text):
        word = m.group()
        is_hyphen = "-" in word
        primary_count += 1
        if is_hyphen:
            concat = word.replace("-", "")
            tokens.append(_stem(concat.lower()))
            if _is_case_candidate(concat):
                tokens.append(concat)
            for part in word.split("-"):
                tokens.append(_stem(part.lower()))
        else:
            tokens.append(_stem(word.lower()))
            if _is_case_candidate(word):
                tokens.append(word)
    return tokens, primary_count


def tokenize(text: str) -> List[str]:
    tokens, _ = tokenize_v1(text)
    if _CASE_TERMS is not None:
        tokens = [t for t in tokens if t == t.lower() or t in _CASE_TERMS]
    return tokens


def set_case_terms(case_terms: set) -> None:
    global _CASE_TERMS
    _CASE_TERMS = case_terms


# ---------------------------------------------------------------------------
# VByte (variable-byte) codec for the compressed index (save_v2/load_v2).
# Non-negative ints, 7 bits per byte, high bit set on the final byte.
# ---------------------------------------------------------------------------
def _vbyte_encode(nums: List[int]) -> bytes:
    out = bytearray()
    for n in nums:
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b)
            else:
                out.append(b | 0x80)
                break
    return bytes(out)


def _vbyte_decode_n(data: bytes, pos: int, count: int) -> Tuple[List[int], int]:
    """Decode `count` VByte numbers from `data` starting at `pos`.
    Returns (numbers, new_pos)."""
    nums = []
    for _ in range(count):
        n = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            n |= (b & 0x7F) << shift
            if b & 0x80:
                break
            shift += 7
        nums.append(n)
    return nums, pos


# Optional compiled accelerator for the postings decode (see
# submission/_vbyte_cpp.cpp + setup.py). If it isn't built, we fall back to
# pure Python — the submission runs either way.
try:
    from submission._vbyte_cpp import decode_all as _cpp_decode_all
except ImportError:
    _cpp_decode_all = None


def _decode_all(blob: bytes, dfs: List[int]):
    """For each term (given its df, in blob order) return (absolute sorted
    doc ordinals, term frequencies). Uses the compiled decoder when present."""
    if _cpp_decode_all is not None:
        return _cpp_decode_all(bytes(blob), dfs)
    out = []
    pos = 0
    for df in dfs:
        gaps, pos = _vbyte_decode_n(blob, pos, df)
        tfs, pos = _vbyte_decode_n(blob, pos, df)
        ords = []
        cur = 0
        for i, g in enumerate(gaps):
            cur = g if i == 0 else cur + g
            ords.append(cur)
        out.append((ords, tfs))
    return out


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.doc_text: Dict[str, str] = (
            {}
        )  # doc_id -> raw text (handy for VSM/debugging)
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """
        corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().
        """
        for doc_id, text in corpus:
            tokens, primary_count = tokenize_v1(text)
            self.doc_len[doc_id] = primary_count
            self.doc_text[doc_id] = text
            for token in tokens:
                if token not in self.postings:
                    self.postings[token] = {}
                if doc_id not in self.postings[token]:
                    self.postings[token][doc_id] = 0
                self.postings[token][doc_id] += 1

        self.N = len(corpus)
        self.avg_doc_len = sum(self.doc_len.values()) / self.N

    def document_frequency(self, term: str) -> int:
        """
        Number of documents containing `term` at least once.
        """
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """Persist everything document_frequency() / your scorers need to
        `index_dir`, so `load()` can reconstruct this object in a fresh
        process with no memory of `build()` ever having run. Called from
        retrieve.build_index().

        The on-disk byte size of whatever you write here is graded
        directly (assignment Section 7, "index size", relative to the
        class median) — some starting points, roughly in order of effort:
          - json/pickle-dump self.postings etc. directly (works, but
            verbose: repeats every doc_id string per posting).
          - drop self.doc_text if your scorers don't need raw text at
            query time (BM25/VSM only need term-frequency and length
            statistics, not the original documents).
          - delta-encode each postings list's doc-ids (sorted ascending,
            store gaps instead of absolute ids) and varint/byte-pack them,
            instead of a naive JSON list of integers.
        """
        case_terms = set()
        prune = []
        df_ratio_threshold = 0.5
        for term in self.postings:
            if term != term.lower():
                canonical = _STEMMER.stem(term.lower())
                canon_df = self.document_frequency(canonical)
                if canon_df == 0:
                    case_terms.add(term)
                    continue
                ratio = self.document_frequency(term) / canon_df
                if ratio <= df_ratio_threshold:
                    case_terms.add(term)
                else:
                    prune.append(term)
        for term in prune:
            del self.postings[term]

        json_data = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "case_terms": list(case_terms),
        }
        with open(f"{index_dir}/index.json", "w") as f:
            json.dump(json_data, f)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.
        """
        with open(f"{index_dir}/index.json", "r") as f:
            json_data = json.load(f)
        index = cls()
        index.postings = json_data["postings"]
        index.doc_len = json_data["doc_len"]
        index.N = json_data["N"]
        index.avg_doc_len = json_data["avg_doc_len"]
        case_terms = set(json_data.get("case_terms", []))
        set_case_terms(case_terms)
        return index

    # -----------------------------------------------------------------------
    # Compressed persistence (v2): int doc-id mapping + gap encoding + VByte.
    # Same on-disk information as save()/load() above, but a fraction of the
    # bytes. Decompresses to the identical in-memory structure, so BM25/VSM
    # and retrieve() are unchanged. save()/load() (JSON) are kept intact as
    # the readable v0 baseline.
    #
    # Files written to index_dir:
    #   meta.json    - N, avg_doc_len, case_terms
    #   docs.txt     - N lines "doc_id<TAB>doc_len"; line number = ordinal
    #   terms.txt    - vocab lines "term<TAB>df", in postings.bin block order
    #   postings.bin - per term: VByte(gaps of sorted doc ordinals) then
    #                  VByte(term frequencies), blocks concatenated in the
    #                  same order as terms.txt
    # -----------------------------------------------------------------------
    def save_v2(self, index_dir: str) -> None:
        # Same truecase df-ratio gate as save() (kept in sync deliberately).
        case_terms = set()
        prune = []
        for term in self.postings:
            if term != term.lower():
                canon_df = self.document_frequency(_STEMMER.stem(term.lower()))
                if canon_df == 0 or self.document_frequency(term) / canon_df <= 0.5:
                    case_terms.add(term)
                else:
                    prune.append(term)
        for term in prune:
            del self.postings[term]

        # Assign an integer ordinal to every doc (build order).
        docid_to_ord = {}
        with open(f"{index_dir}/docs.txt", "w") as f:
            for doc_id, dl in self.doc_len.items():
                docid_to_ord[doc_id] = len(docid_to_ord)
                f.write(f"{doc_id}\t{dl}\n")

        term_lines = []
        blob = bytearray()
        for term, plist in self.postings.items():
            items = sorted((docid_to_ord[d], tf) for d, tf in plist.items())
            ords = [o for o, _ in items]
            tfs = [tf for _, tf in items]
            gaps = [ords[0]] + [ords[i] - ords[i - 1] for i in range(1, len(ords))]
            blob += _vbyte_encode(gaps)
            blob += _vbyte_encode(tfs)
            term_lines.append(f"{term}\t{len(ords)}")

        with open(f"{index_dir}/terms.txt", "w") as f:
            f.write("\n".join(term_lines))
        with open(f"{index_dir}/postings.bin", "wb") as f:
            f.write(bytes(blob))
        with open(f"{index_dir}/meta.json", "w") as f:
            json.dump(
                {"N": self.N, "avg_doc_len": self.avg_doc_len, "case_terms": list(case_terms)},
                f,
            )

    @classmethod
    def load_v2(cls, index_dir: str) -> "InvertedIndex":
        index = cls()
        with open(f"{index_dir}/meta.json") as f:
            meta = json.load(f)
        index.N = meta["N"]
        index.avg_doc_len = meta["avg_doc_len"]
        set_case_terms(set(meta.get("case_terms", [])))

        ord_to_docid = []
        with open(f"{index_dir}/docs.txt") as f:
            for line in f:
                doc_id, dl = line.rstrip("\n").split("\t")
                ord_to_docid.append(doc_id)
                index.doc_len[doc_id] = int(dl)

        with open(f"{index_dir}/postings.bin", "rb") as f:
            blob = f.read()
        with open(f"{index_dir}/terms.txt") as f:
            terms_txt = f.read()

        terms = []
        dfs = []
        for line in terms_txt.split("\n") if terms_txt else []:
            term, df_s = line.split("\t")
            terms.append(term)
            dfs.append(int(df_s))

        for term, (ords, tfs) in zip(terms, _decode_all(blob, dfs)):
            index.postings[term] = {
                ord_to_docid[o]: tf for o, tf in zip(ords, tfs)
            }
        return index
