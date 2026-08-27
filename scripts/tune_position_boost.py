"""
scripts/tune_position_boost.py — preliminary check: does boosting a term's
early-document occurrences help BM25?

Motivation: BM25F-style literature weights title-field term frequency above
body TF. We have no title field, so we approximate "importance by position":
occurrences within the first EARLY_ZONE word-positions of a document count
`boost` times more toward TF.

Data model (cheap): we do NOT store full positions. Per posting we store
just (tf_early, tf_rest) — how many of a term's occurrences fell in the
first EARLY_ZONE words vs after. The per-document early/rest word counts
derive from the document length L: early = min(EARLY_ZONE, L), rest = L -
early.

Scoring (BM25F-style single field, self-consistent so boost=1.0 reproduces
the current BM25 exactly):

    weighted_tf(t,d)  = boost * tf_early + tf_rest
    weighted_len(d)   = boost * min(Z, L) + max(0, L - Z)
    avgdl'            = mean weighted_len over the corpus
    score  = sum_t idf(t) * wtf*(k1+1) / (wtf + k1*(1 - b + b*wlen/avgdl'))

idf and df are position-independent, so they don't change with boost.

We sweep boost, score BM25-ALONE nDCG@10 (deliberately isolating the
position signal from RRF fusion), and plot nDCG vs boost. boost=1.0 is the
built-in sanity check: it must equal the current BM25 dev score (~0.590).

Results (JSON + PNG) are written for the report regardless of outcome.

Usage:
    python -m scripts.tune_position_boost \
        --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv \
        --qrels data/full/qrels_dev.txt \
        --out-json runs/position_boost.json \
        --out-plot runs/position_boost.png
"""

import argparse
import json
import math

from submission.corpus_utils import load_corpus
from submission.indexer import (
    _WORD_RE,
    _STEMMER,
    _is_case_candidate,
    tokenize,
    set_case_terms,
)
from harness.metrics import ndcg_at_k

EARLY_ZONE = 20
BOOSTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
BM25_K1 = 1.8
BM25_B = 0.6
DF_RATIO = 0.5  # same case-term gate the real index uses


def emit_with_positions(text):
    """Yield (token, primary_position) for a document, mirroring
    indexer.tokenize_v1's emission but tagging each token with the 0-based
    index of the word it came from."""
    for pos, m in enumerate(_WORD_RE.finditer(text)):
        word = m.group()
        if "-" in word:
            concat = word.replace("-", "")
            yield _STEMMER.stem(concat.lower()), pos
            if _is_case_candidate(concat):
                yield concat, pos
            for part in word.split("-"):
                yield _STEMMER.stem(part.lower()), pos
        else:
            yield _STEMMER.stem(word.lower()), pos
            if _is_case_candidate(word):
                yield word, pos


def read_queries(path):
    out = []
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            out.append((qid, text))
    return out


def read_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            qid, _, doc_id, rel = line.split()
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def build_position_index(corpus):
    """postings[term][doc] = [tf_early, tf_rest]; doc_len[doc] = L (word count).
    Applies the same df-ratio case-term gate the real index uses so the
    tokenisation matches the current system."""
    postings = {}
    doc_len = {}
    for doc_id, text in corpus:
        n_words = 0
        for token, pos in emit_with_positions(text):
            if pos + 1 > n_words:
                n_words = pos + 1
            bucket = 0 if pos < EARLY_ZONE else 1
            per_doc = postings.setdefault(token, {})
            counts = per_doc.get(doc_id)
            if counts is None:
                counts = [0, 0]
                per_doc[doc_id] = counts
            counts[bucket] += 1
        doc_len[doc_id] = n_words

    N = len(corpus)

    # df-ratio gate on case terms (mirror InvertedIndex.save()).
    case_terms = set()
    prune = []
    for term in postings:
        if term != term.lower():
            canon = _STEMMER.stem(term.lower())
            canon_df = len(postings.get(canon, {}))
            if canon_df == 0:
                case_terms.add(term)
            elif len(postings[term]) / canon_df <= DF_RATIO:
                case_terms.add(term)
            else:
                prune.append(term)
    for term in prune:
        del postings[term]
    set_case_terms(case_terms)
    return postings, doc_len, N


def score_boost(postings, doc_len, N, q_tokens, qrels, qids, boost):
    """Mean nDCG@10 over qids for a given early-token boost."""
    # Weighted document lengths and their mean, for this boost.
    wlen = {}
    for d, L in doc_len.items():
        early = min(EARLY_ZONE, L)
        wlen[d] = boost * early + (L - early)
    avgdl = sum(wlen.values()) / N

    total = 0.0
    for q in qids:
        scores = {}
        for tok in q_tokens[q]:
            plist = postings.get(tok)
            if not plist:
                continue
            df = len(plist)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for d, (tf_e, tf_r) in plist.items():
                wtf = boost * tf_e + tf_r
                denom = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wlen[d] / avgdl)
                scores[d] = scores.get(d, 0.0) + idf * (wtf * (BM25_K1 + 1)) / denom
        ranked = [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]
        total += ndcg_at_k(ranked, qrels[q], k=10)
    return total / len(qids)


def make_plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [b for b, _ in results]
    ys = [n for _, n in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o")
    ax.axhline(ys[0], ls="--", color="gray", alpha=0.6,
               label=f"boost=1.0 baseline ({ys[0]:.4f})")
    ax.set_xlabel(f"early-token boost (first {EARLY_ZONE} words)")
    ax.set_ylabel("nDCG@10 (dev, BM25 alone)")
    ax.set_title("Position-weighted TF: nDCG@10 vs early-token boost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out-json", default="runs/position_boost.json")
    ap.add_argument("--out-plot", default="runs/position_boost.png")
    args = ap.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} docs. Building position-aware index once...")
    postings, doc_len, N = build_position_index(corpus)
    print(f"  vocab={len(postings)}")

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    qids = [q for q, _ in queries if q in qrels]
    q_tokens = {q: tokenize(text) for q, text in queries}

    results = []
    for boost in BOOSTS:
        ndcg = score_boost(postings, doc_len, N, q_tokens, qrels, qids, boost)
        results.append((boost, ndcg))
        print(f"  boost={boost:<4}  nDCG@10={ndcg:.4f}")

    base = results[0][1]
    best_boost, best_ndcg = max(results, key=lambda x: x[1])
    print(f"\nboost=1.0 baseline: {base:.4f}  (sanity: should match current BM25 ~0.590)")
    print(f"best: boost={best_boost}  nDCG@10={best_ndcg:.4f}  (delta {best_ndcg-base:+.4f})")

    with open(args.out_json, "w") as f:
        json.dump({
            "early_zone": EARLY_ZONE,
            "k1": BM25_K1, "b": BM25_B,
            "results": {str(b): n for b, n in results},
            "baseline_boost1": base,
            "best_boost": best_boost,
            "best_ndcg": best_ndcg,
        }, f, indent=2)
    print(f"Wrote {args.out_json}")
    make_plot(results, args.out_plot)


if __name__ == "__main__":
    main()
