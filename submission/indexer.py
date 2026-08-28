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

import os
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


# nltk.corpus.stopwords.words("english") (198 words, NLTK 3.9.x), inlined
# rather than imported: `pip install -r requirements.txt` installs the nltk
# package but not its corpora, so the import raises LookupError on a fresh
# grading container, and build_index() is timed so it must not download.
# Dev effect: nDCG@10 0.6419 -> 0.6607 pre-RM3, index 84MB -> 69MB, query
# latency halved.
_NLTK_STOPWORDS = [
    "a", "about", "above", "after", "again", "against", "ain", "all",
    "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "couldn", "couldn't", "d", "did", "didn",
    "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
    "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
    "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
    "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me",
    "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn",
    "shouldn't", "should've", "so", "some", "such", "t", "than", "that",
    "that'll", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve",
    "very", "was", "wasn", "wasn't", "we", "we'd", "we'll", "we're",
    "were", "weren", "weren't", "we've", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won", "won't",
    "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "your", "you're",
    "yours", "yourself", "yourselves", "you've"
]

# _emit_word applies the stoplist to the WHOLE word, so splitting a compound
# used to leak function words straight into the postings: "state-of-the-art"
# emitted "of"/"the", leaving df(to)=4473, df(of)=2404 in a supposedly
# stopword-free index. Parts are now stoplisted too. Peak nDCG@10 by corpus,
# before -> after: COVID .6836 -> .6829, nfcorpus .3353 -> .3355,
# fiqa .2230 -> .2248, antique .3315 -> .3318.
#
# SRE_HYPH_NUM=1 additionally drops parts that are pure digits or single chars
# ("19" from covid-19, "2" from SARS-CoV-2). Measured NEGATIVE on dev
# (-0.0025) and null off-domain, so it is off by default.
_DROP_DEGENERATE_PARTS = os.environ.get("SRE_HYPH_NUM") == "1" or \
    os.environ.get("SRE_HYPH") == "1"

# Stemmed, because the index stores stems ("having" -> "have").
_STEMMED_STOPWORDS = frozenset(_stem(w) for w in _NLTK_STOPWORDS)


def _iter_words(text: str):
    """Yield raw word strings from `text`."""
    for m in _WORD_RE.finditer(text):
        yield m.group()


def _emit_word(word: str, tokens: List[str]) -> int:
    """v1 emission rules for one word. Returns primary tokens emitted (0 if a
    stopword), so doc_len excludes them."""
    if "-" in word:
        concat = word.replace("-", "")
        stem = _stem(concat.lower())
        if stem in _STEMMED_STOPWORDS:
            return 0
        tokens.append(stem)
        if _is_case_candidate(concat):
            tokens.append(concat)
        for part in word.split("-"):
            low = part.lower()
            if _DROP_DEGENERATE_PARTS and (len(low) < 2 or low.isdigit()):
                continue
            pstem = _stem(low)
            if pstem in _STEMMED_STOPWORDS:
                continue
            tokens.append(pstem)
        return 1
    stem = _stem(word.lower())
    if stem in _STEMMED_STOPWORDS:
        return 0
    tokens.append(stem)
    if _is_case_candidate(word):
        tokens.append(word)
    return 1


def tokenize_v2(text: str) -> Tuple[List[str], int]:
    """Shipped tokenizer: v1's emission rules plus stopword removal."""
    tokens: List[str] = []
    primary = 0
    for word in _iter_words(text):
        primary += _emit_word(word, tokens)
    return tokens, primary


def tokenize_doc(text: str) -> Tuple[List[str], int]:
    """Tokeniser used by the index builder AND the query path."""
    return tokenize_v2(text)


def tokenize(text: str) -> List[str]:
    tokens, _ = tokenize_doc(text)
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
            tokens, primary_count = tokenize_doc(text)
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

        # Forward index (doc -> [(term_ord, tf)]) for RM3 relevance models.
        # term_ord is the term's position in the self.postings iteration order,
        # which is exactly terms.txt order == the ids NativeIndex assigns. Same
        # per-doc format as _spimi_cpp.write_forward: VByte(count) then gap-coded
        # term-ordinal gaps then tfs, one record per doc in docs.txt order.
        forward = {d: [] for d in self.doc_len}
        for term_ord, (_term, plist) in enumerate(self.postings.items()):
            for d, tf in plist.items():
                forward[d].append((term_ord, tf))
        fblob = bytearray()
        for doc_id in self.doc_len:  # docs.txt order
            items = sorted(forward[doc_id])  # ascending term_ord
            ords = [o for o, _ in items]
            tfs = [t for _, t in items]
            gaps = [ords[0]] + [ords[i] - ords[i - 1] for i in range(1, len(ords))] if ords else []
            fblob += _vbyte_encode([len(ords)])
            fblob += _vbyte_encode(gaps)
            fblob += _vbyte_encode(tfs)
        with open(f"{index_dir}/forward.bin", "wb") as f:
            f.write(bytes(fblob))

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
