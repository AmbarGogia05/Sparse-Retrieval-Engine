// submission/_stem_cpp.cpp — native English Snowball (Porter2) stemmer.
//
// A line-for-line C++ port of nltk.stem.snowball.EnglishStemmer.stem (the exact
// version installed in this venv), written from scratch. It is validated to be
// BYTE-IDENTICAL to that stemmer on the full corpus vocab (see the validation
// harness), so swapping it in changes no stem, no posting, and no ranking — it
// is a pure speed win. Stemming is ~95% of build time and also runs per query
// token; nltk's Snowball is pure Python, this is C++.
//
// The port mirrors nltk's unusual model exactly: `word`, `r1`, and `r2` are
// three independent strings kept truncated in parallel (r1/r2 are the R1/R2
// regions as literal substrings), including nltk's quirks (e.g. the r2="e"
// fallbacks in step 2). Do not "clean these up" — fidelity to nltk is the
// whole point; any divergence would silently change the index.
//
// Optional + drop-in: indexer._stem uses it if built, else falls back to nltk.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

namespace {

const std::string VOWELS = "aeiouy";
const std::string DOUBLES[] = {"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"};
const std::string LI_ENDING = "cdeghkmnrt";

inline bool is_vowel(char c) { return VOWELS.find(c) != std::string::npos; }

inline bool ends(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
inline bool ends_any(const std::string &s, const std::string *arr, int n) {
    for (int i = 0; i < n; i++) if (ends(s, arr[i])) return true;
    return false;
}
// Python s[:-n]: drop the last n chars, or "" if n >= len.
inline std::string chop(const std::string &s, size_t n) {
    return s.size() > n ? s.substr(0, s.size() - n) : std::string();
}
// suffix_replace(s, old, new): s[:-len(old)] + new.
inline std::string srep(const std::string &s, size_t oldlen, const std::string &nw) {
    return chop(s, oldlen) + nw;
}
// Python negative index s[-i] (1-based from end); '\0' if out of range.
inline char at(const std::string &s, size_t i) {
    return s.size() >= i ? s[s.size() - i] : '\0';
}

const std::unordered_map<std::string, std::string> &special_words() {
    static const std::unordered_map<std::string, std::string> m = {
        {"skis", "ski"}, {"skies", "sky"}, {"dying", "die"}, {"lying", "lie"},
        {"tying", "tie"}, {"idly", "idl"}, {"gently", "gentl"}, {"ugly", "ugli"},
        {"early", "earli"}, {"only", "onli"}, {"singly", "singl"}, {"sky", "sky"},
        {"news", "news"}, {"howe", "howe"}, {"atlas", "atlas"}, {"cosmos", "cosmos"},
        {"bias", "bias"}, {"andes", "andes"}, {"inning", "inning"},
        {"innings", "inning"}, {"outing", "outing"}, {"outings", "outing"},
        {"canning", "canning"}, {"cannings", "canning"}, {"herring", "herring"},
        {"herrings", "herring"}, {"earring", "earring"}, {"earrings", "earring"},
        {"proceed", "proceed"}, {"proceeds", "proceed"}, {"proceeded", "proceed"},
        {"proceeding", "proceed"}, {"exceed", "exceed"}, {"exceeds", "exceed"},
        {"exceeded", "exceed"}, {"exceeding", "exceed"}, {"succeed", "succeed"},
        {"succeeds", "succeed"}, {"succeeded", "succeed"}, {"succeeding", "succeed"},
    };
    return m;
}

void r1r2_standard(const std::string &word, std::string &r1, std::string &r2) {
    r1.clear(); r2.clear();
    for (size_t i = 1; i < word.size(); i++)
        if (!is_vowel(word[i]) && is_vowel(word[i - 1])) { r1 = word.substr(i + 1); break; }
    for (size_t i = 1; i < r1.size(); i++)
        if (!is_vowel(r1[i]) && is_vowel(r1[i - 1])) { r2 = r1.substr(i + 1); break; }
}

std::string stem_one(const std::string &input) {
    std::string word = input;
    for (char &c : word) if (c >= 'A' && c <= 'Z') c += 32;  // lower()

    if (word.size() <= 2) return word;
    {
        auto it = special_words().find(word);
        if (it != special_words().end()) return it->second;
    }
    // (apostrophe normalisation omitted: tokenizer strips apostrophes)

    if (!word.empty() && word[0] == 'y') word[0] = 'Y';
    for (size_t i = 1; i < word.size(); i++)
        if (is_vowel(word[i - 1]) && word[i] == 'y') word[i] = 'Y';

    std::string r1, r2;
    if (word.rfind("gener", 0) == 0 || word.rfind("arsen", 0) == 0 ||
        word.rfind("commun", 0) == 0) {
        r1 = (word.rfind("commun", 0) == 0) ? word.substr(6) : word.substr(5);
        for (size_t i = 1; i < r1.size(); i++)
            if (!is_vowel(r1[i]) && is_vowel(r1[i - 1])) { r2 = r1.substr(i + 1); break; }
    } else {
        r1r2_standard(word, r1, r2);
    }

    // STEP 0
    {
        const std::string s0[] = {"'s'", "'s", "'"};
        for (const auto &suf : s0)
            if (ends(word, suf)) {
                word = chop(word, suf.size()); r1 = chop(r1, suf.size()); r2 = chop(r2, suf.size());
                break;
            }
    }

    // STEP 1a
    {
        const std::string s1a[] = {"sses", "ied", "ies", "us", "ss", "s"};
        for (const auto &suf : s1a) {
            if (!ends(word, suf)) continue;
            if (suf == "sses") {
                word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
            } else if (suf == "ied" || suf == "ies") {
                if (chop(word, suf.size()).size() > 1) {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                } else {
                    word = chop(word, 1); r1 = chop(r1, 1); r2 = chop(r2, 1);
                }
            } else if (suf == "s") {
                bool vf = false;
                std::string pre = chop(word, 2);
                for (char c : pre) if (is_vowel(c)) { vf = true; break; }
                if (vf) { word = chop(word, 1); r1 = chop(r1, 1); r2 = chop(r2, 1); }
            }
            break;
        }
    }

    // STEP 1b
    {
        const std::string s1b[] = {"eedly", "ingly", "edly", "eed", "ing", "ed"};
        for (const auto &suf : s1b) {
            if (!ends(word, suf)) continue;
            if (suf == "eed" || suf == "eedly") {
                if (ends(r1, suf)) {
                    word = srep(word, suf.size(), "ee");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ee") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ee") : std::string();
                }
            } else {
                bool vf = false;
                std::string pre = chop(word, suf.size());
                for (char c : pre) if (is_vowel(c)) { vf = true; break; }
                if (vf) {
                    word = chop(word, suf.size()); r1 = chop(r1, suf.size()); r2 = chop(r2, suf.size());
                    if (ends(word, "at") || ends(word, "bl") || ends(word, "iz")) {
                        word += "e"; r1 += "e";
                        if (word.size() > 5 || r1.size() >= 3) r2 += "e";
                    } else if (ends_any(word, DOUBLES, 9)) {
                        word = chop(word, 1); r1 = chop(r1, 1); r2 = chop(r2, 1);
                    } else if ((r1.empty() && word.size() >= 3 &&
                                !is_vowel(at(word, 1)) &&
                                std::string("wxY").find(at(word, 1)) == std::string::npos &&
                                is_vowel(at(word, 2)) && !is_vowel(at(word, 3))) ||
                               (r1.empty() && word.size() == 2 &&
                                is_vowel(word[0]) && !is_vowel(word[1]))) {
                        word += "e";
                        if (!r1.empty()) r1 += "e";
                        if (!r2.empty()) r2 += "e";
                    }
                }
            }
            break;
        }
    }

    // STEP 1c
    if (word.size() > 2 && (at(word, 1) == 'y' || at(word, 1) == 'Y') &&
        !is_vowel(at(word, 2))) {
        word = chop(word, 1) + "i";
        r1 = (r1.size() >= 1) ? chop(r1, 1) + "i" : std::string();
        r2 = (r2.size() >= 1) ? chop(r2, 1) + "i" : std::string();
    }

    // STEP 2
    {
        const std::string s2[] = {"ization", "ational", "fulness", "ousness",
            "iveness", "tional", "biliti", "lessli", "entli", "ation", "alism",
            "aliti", "ousli", "iviti", "fulli", "enci", "anci", "abli", "izer",
            "ator", "alli", "bli", "ogi", "li"};
        for (const auto &suf : s2) {
            if (!ends(word, suf)) continue;
            if (ends(r1, suf)) {
                if (suf == "tional") {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                } else if (suf == "enci" || suf == "anci" || suf == "abli") {
                    word = chop(word, 1) + "e";
                    r1 = (r1.size() >= 1) ? chop(r1, 1) + "e" : std::string();
                    r2 = (r2.size() >= 1) ? chop(r2, 1) + "e" : std::string();
                } else if (suf == "entli") {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                } else if (suf == "izer" || suf == "ization") {
                    word = srep(word, suf.size(), "ize");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ize") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ize") : std::string();
                } else if (suf == "ational" || suf == "ation" || suf == "ator") {
                    word = srep(word, suf.size(), "ate");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ate") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ate") : std::string("e");
                } else if (suf == "alism" || suf == "aliti" || suf == "alli") {
                    word = srep(word, suf.size(), "al");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "al") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "al") : std::string();
                } else if (suf == "fulness") {
                    word = chop(word, 4); r1 = chop(r1, 4); r2 = chop(r2, 4);
                } else if (suf == "ousli" || suf == "ousness") {
                    word = srep(word, suf.size(), "ous");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ous") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ous") : std::string();
                } else if (suf == "iveness" || suf == "iviti") {
                    word = srep(word, suf.size(), "ive");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ive") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ive") : std::string("e");
                } else if (suf == "biliti" || suf == "bli") {
                    word = srep(word, suf.size(), "ble");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ble") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ble") : std::string();
                } else if (suf == "ogi" && at(word, 4) == 'l') {
                    word = chop(word, 1); r1 = chop(r1, 1); r2 = chop(r2, 1);
                } else if (suf == "fulli" || suf == "lessli") {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                } else if (suf == "li" &&
                           LI_ENDING.find(at(word, 3)) != std::string::npos) {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                }
            }
            break;
        }
    }

    // STEP 3
    {
        const std::string s3[] = {"ational", "tional", "alize", "icate", "iciti",
            "ative", "ical", "ness", "ful"};
        for (const auto &suf : s3) {
            if (!ends(word, suf)) continue;
            if (ends(r1, suf)) {
                if (suf == "tional") {
                    word = chop(word, 2); r1 = chop(r1, 2); r2 = chop(r2, 2);
                } else if (suf == "ational") {
                    word = srep(word, suf.size(), "ate");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ate") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ate") : std::string();
                } else if (suf == "alize") {
                    word = chop(word, 3); r1 = chop(r1, 3); r2 = chop(r2, 3);
                } else if (suf == "icate" || suf == "iciti" || suf == "ical") {
                    word = srep(word, suf.size(), "ic");
                    r1 = (r1.size() >= suf.size()) ? srep(r1, suf.size(), "ic") : std::string();
                    r2 = (r2.size() >= suf.size()) ? srep(r2, suf.size(), "ic") : std::string();
                } else if (suf == "ful" || suf == "ness") {
                    word = chop(word, suf.size()); r1 = chop(r1, suf.size()); r2 = chop(r2, suf.size());
                } else if (suf == "ative" && ends(r2, suf)) {
                    word = chop(word, 5); r1 = chop(r1, 5); r2 = chop(r2, 5);
                }
            }
            break;
        }
    }

    // STEP 4
    {
        const std::string s4[] = {"ement", "ance", "ence", "able", "ible", "ment",
            "ant", "ent", "ism", "ate", "iti", "ous", "ive", "ize", "ion", "al",
            "er", "ic"};
        for (const auto &suf : s4) {
            if (!ends(word, suf)) continue;
            if (ends(r2, suf)) {
                if (suf == "ion") {
                    if (std::string("st").find(at(word, 4)) != std::string::npos) {
                        word = chop(word, 3); r1 = chop(r1, 3); r2 = chop(r2, 3);
                    }
                } else {
                    word = chop(word, suf.size()); r1 = chop(r1, suf.size()); r2 = chop(r2, suf.size());
                }
            }
            break;
        }
    }

    // STEP 5
    if (ends(r2, "l") && at(word, 2) == 'l') {
        word = chop(word, 1);
    } else if (ends(r2, "e")) {
        word = chop(word, 1);
    } else if (ends(r1, "e")) {
        if (word.size() >= 4 &&
            (is_vowel(at(word, 2)) ||
             std::string("wxY").find(at(word, 2)) != std::string::npos ||
             !is_vowel(at(word, 3)) || is_vowel(at(word, 4)))) {
            word = chop(word, 1);
        }
    }

    for (char &c : word) if (c == 'Y') c = 'y';
    return word;
}

std::vector<std::string> stem_many(const std::vector<std::string> &words) {
    std::vector<std::string> out;
    out.reserve(words.size());
    for (const auto &w : words) out.push_back(stem_one(w));
    return out;
}

// --------------------------------------------------------------------------
// Native tokenizer — a byte-for-byte port of indexer.tokenize_v2 (the shipped
// document/query tokenizer), fused with the stemmer so a document is tokenised
// AND stemmed in one C++ pass with no Python/pybind per-token overhead. This is
// what makes the corpus loop cheap: the old path crossed into Python for every
// token and did a Python dict lookup per occurrence. See indexer.py for the
// rule rationale (hyphen splitting, truecase candidates, stopword removal).
//
// Caching: g_stem_cache memoises lower(word) -> stem, replacing the Python
// _STEM_CACHE for the C++ path (each unique word is stemmed once). Both globals
// are per-process (build runs one Pool worker per process) and touched only by
// the single tokenising thread, so no locking is needed.
// --------------------------------------------------------------------------

std::unordered_set<std::string> g_stopword_stems;
std::unordered_map<std::string, std::string> g_stem_cache;

void set_stopword_stems(const std::vector<std::string> &stems) {
    g_stopword_stems.clear();
    for (const auto &s : stems) g_stopword_stems.insert(s);
}

inline bool is_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
inline std::string lower_ascii(const std::string &s) {
    std::string o = s;
    for (char &c : o) if (c >= 'A' && c <= 'Z') c += 32;
    return o;
}
inline bool has_internal_upper(const std::string &s) {
    for (size_t i = 1; i < s.size(); i++) if (s[i] >= 'A' && s[i] <= 'Z') return true;
    return false;
}
inline bool is_case_candidate(const std::string &w) {
    return w.size() >= 2 && has_internal_upper(w);
}
const std::string &stem_cached(const std::string &lc) {
    auto it = g_stem_cache.find(lc);
    if (it != g_stem_cache.end()) return it->second;
    return g_stem_cache.emplace(lc, stem_one(lc)).first->second;
}
inline bool is_stopword(const std::string &stem) {
    return g_stopword_stems.find(stem) != g_stopword_stems.end();
}

// Emit one matched word (may contain '-'); returns 1 if a primary token was
// emitted, 0 if the whole word was a stopword. Mirrors indexer._emit_word with
// _DROP_DEGENERATE_PARTS off (the shipped default).
int emit_word(const std::string &word, std::vector<std::string> &tokens) {
    if (word.find('-') != std::string::npos) {
        std::string concat;
        concat.reserve(word.size());
        for (char c : word) if (c != '-') concat += c;
        const std::string &stem = stem_cached(lower_ascii(concat));
        if (is_stopword(stem)) return 0;
        tokens.push_back(stem);
        if (is_case_candidate(concat)) tokens.push_back(concat);
        size_t start = 0;
        while (start <= word.size()) {
            size_t dash = word.find('-', start);
            std::string part = word.substr(start, dash == std::string::npos
                                                       ? std::string::npos
                                                       : dash - start);
            const std::string &ps = stem_cached(lower_ascii(part));
            if (!is_stopword(ps)) tokens.push_back(ps);
            if (dash == std::string::npos) break;
            start = dash + 1;
        }
        return 1;
    }
    const std::string &stem = stem_cached(lower_ascii(word));
    if (is_stopword(stem)) return 0;
    tokens.push_back(stem);
    if (is_case_candidate(word)) tokens.push_back(word);
    return 1;
}

// Scan words matching the regex [A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|[A-Za-z0-9]+
// (ASCII-only, exactly like Python's _WORD_RE) and tokenise each. Returns
// (tokens, primary_count) where primary_count is the doc length.
std::pair<std::vector<std::string>, int> tokenize_doc(const std::string &text) {
    std::vector<std::string> tokens;
    int primary = 0;
    size_t n = text.size(), i = 0;
    while (i < n) {
        if (!is_alnum(text[i])) { i++; continue; }
        size_t start = i;
        while (i < n && is_alnum(text[i])) i++;
        // Extend across "-run" groups only when '-' is followed by an alnum.
        while (i + 1 < n && text[i] == '-' && is_alnum(text[i + 1])) {
            i++;  // consume '-'
            while (i < n && is_alnum(text[i])) i++;
        }
        primary += emit_word(text.substr(start, i - start), tokens);
    }
    return {tokens, primary};
}

}  // namespace

PYBIND11_MODULE(_stem_cpp, m) {
    m.doc() = "Native English Snowball (Porter2) stemmer + fused tokenizer.";
    m.def("stem", &stem_one, py::arg("word"));
    m.def("stem_many", &stem_many, py::arg("words"));
    m.def("set_stopword_stems", &set_stopword_stems, py::arg("stems"));
    m.def("tokenize_doc", &tokenize_doc, py::arg("text"));
}
