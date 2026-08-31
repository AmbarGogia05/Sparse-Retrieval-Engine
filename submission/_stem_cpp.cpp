// submission/_stem_cpp.cpp — Python bindings for the shared Tokenizer (Porter2
// stemmer + fused tokenizer, defined in tokenizer.cpp / tokenizer.h).
//
// This module is the QUERY-path entry point: indexer._stem calls stem(), and
// indexer.tokenize_doc calls tokenize_doc() (returning a Python list, which the
// query path then filters). The BUILD path uses _spimi_cpp, which embeds its own
// Tokenizer so tokens never cross into Python. A single module-global Tokenizer
// holds the stopword set + stem cache for this (single-threaded) query process.
//
// Optional + drop-in: indexer._stem/​tokenize_doc use it if built, else fall
// back to the pure-Python nltk path.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tokenizer.h"

namespace py = pybind11;

namespace {
Tokenizer g_tok;

std::string stem(const std::string &word) { return Tokenizer::stem(word); }

void set_stopword_stems(const std::vector<std::string> &stems) {
    g_tok.set_stopword_stems(stems);
}

std::pair<std::vector<std::string>, int> tokenize_doc(const std::string &text) {
    return g_tok.tokenize_doc(text);
}
}  // namespace

PYBIND11_MODULE(_stem_cpp, m) {
    m.doc() = "Native English Snowball (Porter2) stemmer + fused tokenizer.";
    m.def("stem", &stem, py::arg("word"));
    m.def("set_stopword_stems", &set_stopword_stems, py::arg("stems"));
    m.def("tokenize_doc", &tokenize_doc, py::arg("text"));
}
