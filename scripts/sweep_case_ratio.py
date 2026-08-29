"""
scripts/sweep_case_ratio.py — sweep the truecase df-ratio gate on the dev set.

The gate (indexer.save_v2 / _spimi_cpp.finalize) keeps a case-preserved term
(e.g. "mRNA", "COVID") only if df(cased)/df(lowercase-stem) <= ratio, else it is
pruned from the index. The threshold changes the INDEX (which case terms enter
postings + meta.json), so every ratio needs a full rebuild — but only the
parent's finalize uses the ratio (workers just collect all case terms), so we
drive it by setting retrieve._CASE_DF_RATIO before each build_index().

Finding (2026-08-29): nDCG@10 is flat within the 50-query noise floor across the
whole 0.0..1.0 range and non-monotonic — 0.5 (the shipped, never-tuned default)
is a coincidental bump, not special. The only real signal is index size (a
monotonic ~3MB swing). Folding in index_size_score, lower ratios score slightly
higher (size penalty > noise-level nDCG gain). See current-competition-entry
memory for the table. Kept 0.5 (changing chases noise + held-out risk).

Usage:
    python -m scripts.sweep_case_ratio \
        --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv \
        --qrels data/full/qrels_dev.txt
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time

from submission import retrieve as R
from harness.metrics import evaluate_run
from harness.run_harness import read_qrels, read_queries

RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _index_bytes(d):
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, "*"))
               if os.path.isfile(p))


def _n_case_terms(d):
    with open(os.path.join(d, "meta.json")) as f:
        return len(json.load(f).get("case_terms", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/full/corpus.jsonl")
    ap.add_argument("--queries", default="data/full/queries_dev.tsv")
    ap.add_argument("--qrels", default="data/full/qrels_dev.txt")
    ap.add_argument("--ratios", nargs="*", type=float, default=RATIOS)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    queries = read_queries(args.queries)
    tmp = tempfile.mkdtemp(prefix="case_sweep_")
    try:
        print(f"{'ratio':>6} {'nDCG@10':>8} {'index_MB':>9} {'#case_terms':>12} {'build_s':>8}")
        for ratio in args.ratios:
            d = os.path.join(tmp, f"idx_{ratio}")
            shutil.rmtree(d, ignore_errors=True)
            R._CASE_DF_RATIO = ratio  # only the parent's finalize reads this
            t = time.perf_counter()
            R.build_index(args.corpus, d)
            bt = time.perf_counter() - t
            R.load_index(d)
            run = {qid: R.retrieve(text, 10) for qid, text in queries}
            ndcg = evaluate_run(run, qrels, k=10)["aggregate"]["ndcg@10"]
            print(f"{ratio:>6.2f} {ndcg:>8.4f} {_index_bytes(d) / 1e6:>9.2f} "
                  f"{_n_case_terms(d):>12} {bt:>8.2f}")
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
