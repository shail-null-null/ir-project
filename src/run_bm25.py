"""Run BM25 on SciFact and NFCorpus, evaluate, cross-check against pytrec_eval,
and print a results table (quality + time + memory).

Run from the project root:   python -m src.run_bm25
"""
import os
import time
import tracemalloc
from statistics import mean

import pytrec_eval

from src.bm25 import BM25
from src.data import filter_queries, load_corpus, load_qrels, load_queries
from src.metrics import average_precision, ndcg_at_k, precision_at_k

DATASETS = ["scifact", "nfcorpus"]
DEPTH = 1000  # documents retrieved per query
METRICS = ["NDCG@5", "NDCG@10", "MAP", "P@10", "R@100", "Judged@10"]


# --- Two extra metrics (to be moved into metrics.py with tests later) -------------
def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """Fraction of all relevant documents (grade >= 1) found in the top k."""
    relevant = {d for d, g in rels.items() if g >= 1}
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def judged_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """Fraction of the top k that has ANY judgment (grade 0 included).
    Low values mean many retrieved documents were never assessed."""
    top = ranked[:k]
    return sum(1 for d in top if d in rels) / len(top) if top else 0.0


# --- Helpers ------------------------------------------------------------------------
def ranking(scores: dict[str, float]) -> list[str]:
    # score descending, ties broken by doc_id descending (trec_eval's convention)
    return sorted(scores, key=lambda d: (scores[d], d), reverse=True)


def per_query_metrics(run, qrels) -> dict[str, dict[str, float]]:
    """Metrics for EVERY query in qrels; queries missing from the run score 0."""
    out = {}
    for qid, rels in qrels.items():
        ranked = ranking(run.get(qid, {}))
        out[qid] = {
            "NDCG@5": ndcg_at_k(ranked, rels, 5),
            "NDCG@10": ndcg_at_k(ranked, rels, 10),
            "MAP": average_precision(ranked, rels),
            "P@10": precision_at_k(ranked, rels, 10),
            "R@100": recall_at_k(ranked, rels, 100),
            "Judged@10": judged_at_k(ranked, rels, 10),
        }
    return out


def pytrec_means(run, qrels) -> dict[str, float]:
    """Same metrics from pytrec_eval, averaged the same way (all qrels queries, missing = 0)."""
    nonempty = {q: docs for q, docs in run.items() if docs}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "ndcg_cut.5,10", "P.10", "recall.100"})
    res = evaluator.evaluate(nonempty)
    names = {"NDCG@5": "ndcg_cut_5", "NDCG@10": "ndcg_cut_10", "MAP": "map",
             "P@10": "P_10", "R@100": "recall_100"}
    return {mine: sum(res.get(q, {}).get(theirs, 0.0) for q in qrels) / len(qrels)
            for mine, theirs in names.items()}


def write_run(run, path: str, tag: str) -> None:
    """TREC run format: qid Q0 doc_id rank score tag"""
    with open(path, "w", encoding="utf-8") as f:
        for qid, scores in run.items():
            for rank, doc_id in enumerate(ranking(scores), start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {scores[doc_id]:.6f} {tag}\n")


# --- Main ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs("runs", exist_ok=True)
    rows = []
    for name in DATASETS:
        base = f"data/{name}"
        corpus = load_corpus(f"{base}/corpus.jsonl")
        qrels = load_qrels(f"{base}/qrels/test.tsv")
        queries = filter_queries(load_queries(f"{base}/queries.jsonl"), qrels)

        # 1. Index (timed without tracing, because tracemalloc slows code down)
        bm25 = BM25()
        t0 = time.perf_counter()
        bm25.index(corpus)
        index_s = time.perf_counter() - t0

        # 2. Index memory: rebuild once under tracemalloc and keep it alive while measuring
        tracemalloc.start()
        probe = BM25()
        probe.index(corpus)
        index_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del probe

        # 3. Search (timed)
        t0 = time.perf_counter()
        run = bm25.search(queries, k=DEPTH)
        search_s = time.perf_counter() - t0
        write_run(run, f"runs/bm25_{name}.trec", "bm25")

        # 4. Evaluate with our evaluator, then cross-check against pytrec_eval
        per_q = per_query_metrics(run, qrels)
        ours = {m: mean(r[m] for r in per_q.values()) for m in METRICS}
        ref = pytrec_means(run, qrels)
        max_diff = max(abs(ours[m] - ref[m]) for m in ref)
        empty = sum(1 for q in qrels if not run.get(q))
        n_postings = sum(len(p) for p in bm25.postings.values())

        print(f"\n=== {name} ===")
        print(f"docs={len(corpus)}  test queries={len(queries)}  queries with no results={empty}")
        print(f"vocabulary={len(bm25.postings)}  postings={n_postings}  avg doc length={bm25.avgdl:.1f} tokens")
        print(f"index: {index_s:.2f} s, {index_bytes / 1e6:.1f} MB   search: {1000 * search_s / len(queries):.2f} ms/query")
        print("ours     :", "  ".join(f"{m}={ours[m]:.4f}" for m in METRICS))
        print("pytrec   :", "  ".join(f"{m}={ref[m]:.4f}" for m in ref))
        print(f"max |ours - pytrec_eval| = {max_diff:.2e}  ->  {'OK' if max_diff < 1e-4 else 'MISMATCH - investigate'}")

        rows.append((name, len(corpus), len(queries), ours, index_s, index_bytes / 1e6,
                     1000 * search_s / len(queries)))

    # Markdown table, ready to paste into the report
    print("\n| Dataset | Docs | Queries | " + " | ".join(METRICS) + " | Index time (s) | Index memory (MB) | Latency (ms/query) |")
    print("|" + "---|" * (len(METRICS) + 6))
    for name, n_docs, n_q, ours, idx_s, idx_mb, ms_q in rows:
        print(f"| {name} | {n_docs} | {n_q} | " + " | ".join(f"{ours[m]:.4f}" for m in METRICS)
              + f" | {idx_s:.2f} | {idx_mb:.1f} | {ms_q:.2f} |")


if __name__ == "__main__":
    main()
