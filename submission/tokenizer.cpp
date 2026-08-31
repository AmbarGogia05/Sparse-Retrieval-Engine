// submission/tokenizer.cpp — definitions for tokenizer.h (shared by _stem_cpp
// and _spimi_cpp). Byte-for-byte port of nltk EnglishStemmer.stem +
// indexer._tokenize_python. See tokenizer.h. Do NOT "clean up" the stemmer's
// three-parallel-string model or nltk's r2="e" quirks — fidelity to nltk is
// what keeps the index byte-identical.

#include "tokenizer.h"

namespace {

const std::string VOWELS = "aeiouy";
const std::string DOUBLES[] = {"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"};
const std::string LI_ENDING = "cdeghkmnrt";

bool is_vowel(char c) { return VOWELS.find(c) != std::string::npos; }

bool ends(const std::string &s, const std::string &suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
bool ends_any(const std::string &s, const std::string *arr, int n) {
    for (int i = 0; i < n; i++) if (ends(s, arr[i])) return true;
    return false;
}
// Python s[:-n]: drop the last n chars, or "" if n >= len.
std::string chop(const std::string &s, size_t n) {
    return s.size() > n ? s.substr(0, s.size() - n) : std::string();
}
// suffix_replace(s, old, new): s[:-len(old)] + new.
std::string srep(const std::string &s, size_t oldlen, const std::string &nw) {
    return chop(s, oldlen) + nw;
}
// Python negative index s[-i] (1-based from end); '\0' if out of range.
char at(const std::string &s, size_t i) {
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

bool is_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
std::string lower_ascii(const std::string &s) {
    std::string o = s;
    for (char &c : o) if (c >= 'A' && c <= 'Z') c += 32;
    return o;
}
bool has_internal_upper(const std::string &s) {
    for (size_t i = 1; i < s.size(); i++) if (s[i] >= 'A' && s[i] <= 'Z') return true;
    return false;
}
bool is_case_candidate(const std::string &w) {
    return w.size() >= 2 && has_internal_upper(w);
}

}  // namespace

std::string Tokenizer::stem(const std::string &word) { return stem_one(word); }

void Tokenizer::set_stopword_stems(const std::vector<std::string> &stems) {
    stopword_stems_.clear();
    for (const auto &s : stems) stopword_stems_.insert(s);
}

bool Tokenizer::is_stopword(const std::string &s) const {
    return stopword_stems_.find(s) != stopword_stems_.end();
}

const std::string &Tokenizer::stem_cached(const std::string &lc) {
    auto it = stem_cache_.find(lc);
    if (it != stem_cache_.end()) return it->second;
    return stem_cache_.emplace(lc, stem_one(lc)).first->second;
}

// Emit one matched word (may contain '-'); returns 1 if a primary token was
// emitted, 0 if the whole word was a stopword. Mirrors indexer._emit_word with
// _DROP_DEGENERATE_PARTS off (the shipped default).
int Tokenizer::emit_word(const std::string &word, std::vector<std::string> &tokens) {
    if (word.find('-') != std::string::npos) {
        std::string concat;
        concat.reserve(word.size());
        for (char c : word) if (c != '-') concat += c;
        const std::string &st = stem_cached(lower_ascii(concat));
        if (is_stopword(st)) return 0;
        tokens.push_back(st);
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
    const std::string &st = stem_cached(lower_ascii(word));
    if (is_stopword(st)) return 0;
    tokens.push_back(st);
    if (is_case_candidate(word)) tokens.push_back(word);
    return 1;
}

// Scan words matching [A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|[A-Za-z0-9]+ (ASCII-only,
// exactly like Python's _WORD_RE) and tokenise each into `out`.
int Tokenizer::tokenize(const std::string &text, std::vector<std::string> &out) {
    out.clear();
    int primary = 0;
    size_t n = text.size(), i = 0;
    while (i < n) {
        if (!is_alnum(text[i])) { i++; continue; }
        size_t start = i;
        while (i < n && is_alnum(text[i])) i++;
        while (i + 1 < n && text[i] == '-' && is_alnum(text[i + 1])) {
            i++;
            while (i < n && is_alnum(text[i])) i++;
        }
        primary += emit_word(text.substr(start, i - start), out);
    }
    return primary;
}

std::pair<std::vector<std::string>, int> Tokenizer::tokenize_doc(const std::string &text) {
    std::vector<std::string> out;
    int p = tokenize(text, out);
    return {std::move(out), p};
}
