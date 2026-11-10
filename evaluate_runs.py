"""Evaluate every run file in runs/ and print the results table.

Run from the project root:   python evaluate_runs.py
"""
import json
import os

from src.data import load_qrels
from src.evaluation import METRICS, mean_metrics, per_query_metrics, pytrec_means, read_run

SYSTEMS = ["bm25", "dense"]
DATASETS = ["scifact", "nfcorpus", "scidocs"]

rows = []
for dataset in DATASETS:
    qrels_path = f"data/{dataset}/qrels/test.tsv"
    if not os.path.exists(qrels_path):
        print(f"skip {dataset}: no qrels at {qrels_path}")
        continue
    qrels = load_qrels(qrels_path)
    for system in SYSTEMS:
        path = f"runs/{system}_{dataset}.trec"
        if not os.path.exists(path):
            print(f"skip {system}/{dataset}: no run file at {path}")
            continue
        run = read_run(path)
        per_q = per_query_metrics(run, qrels)
        means = mean_metrics(per_q)
        ref = pytrec_means(run, qrels)
        diff = max(abs(means[m] - ref[m]) for m in ref)
        empty = sum(1 for q in qrels if not run.get(q))

        # per-query values, kept for the comparison and significance tests
        with open(f"runs/perquery_{system}_{dataset}.json", "w", encoding="utf-8") as f:
            json.dump(per_q, f, indent=1)

        print(f"{system:5s} {dataset:9s} queries={len(qrels)} no-results={empty} "
              f"max|ours-pytrec|={diff:.1e} {'OK' if diff < 1e-4 else 'MISMATCH'}")
        rows.append((dataset, system, means))

print("\n| Dataset | System | " + " | ".join(METRICS) + " |")
print("|" + "---|" * (len(METRICS) + 2))
for dataset, system, means in rows:
    print(f"| {dataset} | {system} | " + " | ".join(f"{means[m]:.4f}" for m in METRICS) + " |")
