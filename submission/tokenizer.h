// submission/tokenizer.h — shared English Snowball (Porter2) stemmer + fused
// tokenizer, used by BOTH _stem_cpp (query path) and _spimi_cpp (build path).
//
// Declarations only; definitions live in tokenizer.cpp, which is compiled into
// each extension module (see setup.py). This lets the build path tokenise a
// document entirely in C++ (SpimiBuilder owns a Tokenizer) so tokens never
// materialise as a Python list. NativeIndex also owns one for its one-call
// retrieve path; the standalone _stem_cpp query helper still returns a list.
//
// The implementation is a byte-for-byte port of nltk's EnglishStemmer.stem and
// indexer._tokenize_python. Do not alter the algorithm without re-validating
// against nltk: build and query tokenization must remain identical.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Snowball (Porter2) stemmer + fused tokenizer. Holds the stopword-stem set and
// a per-instance stem cache; not thread-safe (each build worker/process and the
// query module hold their own instance, all single-threaded).
class Tokenizer {
public:
    // Stateless Porter2 stem of one word (used by the standalone stem binding).
    static std::string stem(const std::string &word);

    // Configure the stopword filter with already-stemmed stopwords.
    void set_stopword_stems(const std::vector<std::string> &stems);

    // Tokenise `text` into `out` (cleared first); returns the primary token
    // count (the doc length). Fused tokenise+stem.
    int tokenize(const std::string &text, std::vector<std::string> &out);

    // Convenience wrapper returning (tokens, primary_count).
    std::pair<std::vector<std::string>, int> tokenize_doc(const std::string &text);

private:
    std::unordered_set<std::string> stopword_stems_;
    std::unordered_map<std::string, std::string> stem_cache_;

    const std::string &stem_cached(const std::string &lc);
    bool is_stopword(const std::string &s) const;
    int emit_word(const std::string &word, std::vector<std::string> &tokens);
};
