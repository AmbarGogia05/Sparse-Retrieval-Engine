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

"""
v1 of the tokenizer, which is a simple regex-based tokenizer that lowercases,
and for hyphenated words, keeps the hyphenated version as well as the 
individual tokens. For example, "SARS-CoV-2" will be tokenized into 
["sars-cov-2", "sars", "cov", "2"].
"""

_HYPHEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    lower = text.lower()
    tokens = []
    for m in _HYPHEN_RE.finditer(lower):
        tokens.append(m.group())  # "sars-cov-2"
    tokens.extend(_TOKEN_RE.findall(lower))  # "sars", "cov", "2" (plus everything else)
    return tokens


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
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
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
        json_data = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
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
        return index
