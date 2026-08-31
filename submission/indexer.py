"""
submission/indexer.py — tokenizer and Python index fallback.

The module defines the shared tokenizer plus the readable `InvertedIndex`
implementation used by the pure-Python scorer path. The competition build
normally uses the optional C++ SPIMI builder, while `save_v2()` and
`load_v2()` define the compact on-disk format used by both native and Python
loaders.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one, so query-time state must round-trip through the
compact `save_v2()` format and its sidecars.
"""

import json
import os
import re
from typing import Dict, List, Tuple

from nltk.stem.snowball import SnowballStemmer

from submission._index_codec import (
    DOCID_MAGIC,
    decode_case_terms as _decode_case_terms,
    decode_docids as _decode_docids,
    decode_postings as _decode_postings,
    decode_terms as _decode_terms,
    decode_vbyte as _vbyte_decode_n,
    encode_case_terms as _encode_case_terms,
    encode_docid as _pack_docid,
    encode_postings as _encode_postings,
    encode_terms as _encode_terms,
    encode_vbyte as _vbyte_encode,
)

_STEMMER = SnowballStemmer("english")

# Native C++ English Snowball (Porter2) stemmer, drop-in for the pure-Python
# nltk one. It is a line-for-line port of this venv's nltk EnglishStemmer,
# validated BYTE-IDENTICAL on 441K words (207K corpus vocab + the system
# dictionary), 0 mismatches — so the index, postings and ranking are unchanged;
# it is a pure speed win (~11x faster raw stemming). SRE_SNOWBALL=1 forces the
# pure-Python path (used to prove the two agree). Build and query both go
# through _stem, so they always use the same stemmer.
_USE_CPP_STEM = os.environ.get("SRE_SNOWBALL") != "1"
_cpp_tokenize = None
_cpp_set_stopword_stems = None
try:
    from submission._stem_cpp import stem as _cpp_stem
    from submission._stem_cpp import tokenize_doc as _cpp_tokenize
    from submission._stem_cpp import set_stopword_stems as _cpp_set_stopword_stems
except ImportError:
    _USE_CPP_STEM = False

# Memoise by raw word so repeated terms are stemmed only once per process. The
# cache wraps whichever stemmer is active, so the C++ path needs no second
# Python-side cache.
_STEM_CACHE = {}


def _stem(word: str) -> str:
    s = _STEM_CACHE.get(word)
    if s is None:
        s = _cpp_stem(word) if _USE_CPP_STEM else _STEMMER.stem(word)
        _STEM_CACHE[word] = s
    return s


_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+|[a-zA-Z0-9]+")


def _has_internal_upper(s: str) -> bool:
    return any(c.isupper() for c in s[1:])


_CASE_TERMS: set = None


def _is_case_candidate(word: str) -> bool:
    return len(word) >= 2 and _has_internal_upper(word)


# nltk.corpus.stopwords.words("english") (198 words, NLTK 3.9.x), inlined
# rather than imported: `pip install -r requirements.txt` installs the nltk
# package but not its corpora, so the import raises LookupError on a fresh
# grading container, and build_index() is timed so it must not download.
# The list is inlined so a fresh grading environment does not need the NLTK
# corpus download during the timed build.
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

# _emit_word applies the stoplist to both a compound word and its emitted
# pieces, so function words from terms such as "state-of-the-art" do not leak
# into the postings. SRE_HYPH_NUM=1 additionally drops parts that are pure
# digits or single characters; it is disabled by default.
_DROP_DEGENERATE_PARTS = os.environ.get("SRE_HYPH_NUM") == "1" or \
    os.environ.get("SRE_HYPH") == "1"

# Stemmed, because the index stores stems ("having" -> "have").
_STEMMED_STOPWORDS = frozenset(_stem(w) for w in _NLTK_STOPWORDS)

# Use the fused C++ tokenizer (tokenize + stem in one native pass) when the
# extension is built and the pure-Python stemmer isn't being forced (SRE_SNOWBALL
# =1) and the experimental degenerate-part drop isn't on (the C++ port implements
# only the shipped default). It is a byte-for-byte port of _tokenize_python, so
# native and Python tokenization remains identical.
_USE_CPP_TOKENIZE = (
    _USE_CPP_STEM and _cpp_tokenize is not None and not _DROP_DEGENERATE_PARTS
)
if _cpp_set_stopword_stems is not None:
    _cpp_set_stopword_stems(list(_STEMMED_STOPWORDS))


def _iter_words(text: str):
    """Yield raw word strings from `text`."""
    for m in _WORD_RE.finditer(text):
        yield m.group()


def _emit_word(word: str, tokens: List[str]) -> int:
    """Append one word's index terms and return its document-length count."""
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


def _tokenize_python(text: str) -> Tuple[List[str], int]:
    """Pure-Python implementation of the shipped tokenizer."""
    tokens: List[str] = []
    primary = 0
    for word in _iter_words(text):
        primary += _emit_word(word, tokens)
    return tokens, primary


def tokenize_doc(text: str) -> Tuple[List[str], int]:
    """Tokeniser used by the index builder AND the query path. Dispatches to the
    fused C++ tokenizer when available."""
    if _USE_CPP_TOKENIZE:
        return _cpp_tokenize(text)
    return _tokenize_python(text)


def tokenize(text: str) -> List[str]:
    tokens, _ = tokenize_doc(text)
    if _CASE_TERMS is not None:
        tokens = [t for t in tokens if t == t.lower() or t in _CASE_TERMS]
    return tokens


def set_case_terms(case_terms: set) -> None:
    global _CASE_TERMS
    _CASE_TERMS = case_terms


def _load_case_terms(index_dir: str, meta: dict) -> set:
    """Read compact case terms, falling back to legacy meta.json."""
    try:
        with open(f"{index_dir}/case_terms.bin", "rb") as f:
            return _decode_case_terms(f.read())
    except FileNotFoundError:
        return set(meta.get("case_terms", []))


class InvertedIndex:
    """Readable Python inverted-index fallback.

    The native builder/loader is preferred when the compiled extensions are
    available; this class remains the compatibility implementation used by
    tests and by the pure-Python scorer path.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
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
            for token in tokens:
                postings = self.postings.setdefault(token, {})
                postings[doc_id] = postings.get(doc_id, 0) + 1

        self.N = len(corpus)
        self.avg_doc_len = sum(self.doc_len.values()) / self.N

    def document_frequency(self, term: str) -> int:
        """Return the number of documents containing `term`."""
        return len(self.postings.get(term, {}))

    # -----------------------------------------------------------------------
    # Compressed persistence (v2): compact document IDs, front-coded terms,
    # Elias-Fano postings, and VByte/bit-packed sidecars.
    #
    # Files written to index_dir:
    #   meta.json    - N, avg_doc_len
    #   docids.bin   - preferred D36 sidecar for 8-character lowercase IDs
    #   docs.txt     - fallback, one verbatim doc ID per line
    #   doclen.bin   - parallel VByte document lengths
    #   terms.txt    - alphabetically ordered front-coded terms and dfs
    #   postings.bin - per term: Elias-Fano ordinals plus optional packed TFs
    #   tf1.bin      - bitmap marking terms whose postings all have tf == 1
    #   case_terms.bin - CT1 front-coded case-sensitive vocabulary
    # -----------------------------------------------------------------------
    def save_v2(self, index_dir: str) -> None:
        # Keep a case-sensitive term only when it is uncommon relative to its
        # lowercase stem; common variants add postings without useful signal.
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
        # Store document lengths separately from IDs. Prefer compact D36 IDs
        # when every ID is an 8-character lowercase base-36 string; otherwise
        # retain the verbatim IDs in docs.txt.
        docid_to_ord = {doc_id: i for i, doc_id in enumerate(self.doc_len)}
        packed_ids = [_pack_docid(doc_id) for doc_id in self.doc_len]
        if all(value is not None for value in packed_ids):
            with open(f"{index_dir}/docids.bin", "wb") as f:
                f.write(DOCID_MAGIC)
                for value in packed_ids:
                    f.write(value)
            if os.path.exists(f"{index_dir}/docs.txt"):
                os.remove(f"{index_dir}/docs.txt")
        else:
            if os.path.exists(f"{index_dir}/docids.bin"):
                os.remove(f"{index_dir}/docids.bin")
            with open(f"{index_dir}/docs.txt", "w") as f:
                for doc_id in self.doc_len:
                    f.write(f"{doc_id}\n")
        with open(f"{index_dir}/doclen.bin", "wb") as f:
            f.write(_vbyte_encode(list(self.doc_len.values())))

        # Alphabetical order matches NativeIndex term IDs and makes adjacent
        # terms suitable for front coding.
        terms_sorted = sorted(self.postings.keys())

        term_meta = []  # (term, df)
        blob = bytearray()
        tf_one = bytearray(b"TF1" + b"\0" * ((len(terms_sorted) + 7) // 8))
        for term in terms_sorted:
            plist = self.postings[term]
            items = sorted((docid_to_ord[d], tf) for d, tf in plist.items())
            ords = [o for o, _ in items]
            tfs = [tf for _, tf in items]
            all_one = all(tf == 1 for tf in tfs)
            if all_one:
                i = len(term_meta)
                tf_one[3 + i // 8] |= 1 << (i % 8)
            blob += _encode_postings(ords, tfs, self.N, all_one)  # Elias-Fano codec
            term_meta.append((term, len(ords)))

        with open(f"{index_dir}/terms.txt", "wb") as f:
            f.write(_encode_terms(term_meta))
        with open(f"{index_dir}/postings.bin", "wb") as f:
            f.write(bytes(blob))
        with open(f"{index_dir}/tf1.bin", "wb") as f:
            f.write(tf_one)

        # No forward.bin is persisted: the forward index (doc -> [(term_ord,
        # tf)]) that RM3 needs is a byte-for-byte redundant transpose of the
        # postings above, and NativeIndex reconstructs it in RAM at load time
        # (see _index_cpp.cpp load()). Persisting it roughly doubled the index
        # size for zero extra information.

        with open(f"{index_dir}/meta.json", "w") as f:
            json.dump({"N": self.N, "avg_doc_len": self.avg_doc_len}, f)
        with open(f"{index_dir}/case_terms.bin", "wb") as f:
            f.write(_encode_case_terms(case_terms))

    @classmethod
    def load_v2(cls, index_dir: str) -> "InvertedIndex":
        """Load the compact v2 index and its sidecars for the Python fallback."""
        index = cls()
        with open(f"{index_dir}/meta.json") as f:
            meta = json.load(f)
        index.N = meta["N"]
        index.avg_doc_len = meta["avg_doc_len"]
        set_case_terms(_load_case_terms(index_dir, meta))

        packed_path = f"{index_dir}/docids.bin"
        if os.path.exists(packed_path):
            with open(packed_path, "rb") as f:
                ord_to_docid = _decode_docids(f.read())
        else:
            ord_to_docid = []
            with open(f"{index_dir}/docs.txt") as f:
                for line in f:
                    ord_to_docid.append(line.rstrip("\n"))
        with open(f"{index_dir}/doclen.bin", "rb") as f:
            lens, _ = _vbyte_decode_n(f.read(), 0, len(ord_to_docid))
        for doc_id, dl in zip(ord_to_docid, lens):
            index.doc_len[doc_id] = dl

        with open(f"{index_dir}/postings.bin", "rb") as f:
            blob = f.read()
        try:
            with open(f"{index_dir}/tf1.bin", "rb") as f:
                tf_one = f.read()
        except FileNotFoundError:
            tf_one = b""
        with open(f"{index_dir}/terms.txt", "rb") as f:
            terms, dfs = _decode_terms(f.read())

        tf_flags = [
            bool(
                tf_one.startswith(b"TF1")
                and len(tf_one) >= 3 + (len(terms) + 7) // 8
                and ((tf_one[3 + i // 8] >> (i % 8)) & 1)
            )
            for i in range(len(terms))
        ]
        pos = 0
        for i, (term, df) in enumerate(zip(terms, dfs)):
            ords, tfs, pos = _decode_postings(
                blob, pos, df, index.N, tf_flags[i]
            )
            index.postings[term] = {
                ord_to_docid[o]: tf for o, tf in zip(ords, tfs)
            }
        return index
