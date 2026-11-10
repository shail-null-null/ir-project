"""Run files and evaluation, shared by every retriever (BM25, dense, ...).

A run is {qid: {doc_id: score}}. Every metric is computed per query, for EVERY query in
the qrels (a query missing from the run scores 0), then averaged.
"""
import pytrec_eval

from src.metrics import (average_precision, judged_at_k, ndcg_at_k, precision_at_k,
                         recall_at_k)

METRICS = ["NDCG@5", "NDCG@10", "MAP", "P@10", "R@100", "Judged@10"]


def ranking(scores: dict[str, float]) -> list[str]:
    """Doc IDs best first; ties broken by doc_id descending (trec_eval's convention)."""
    return sorted(scores, key=lambda d: (scores[d], d), reverse=True)


def write_run(run: dict[str, dict[str, float]], path: str, tag: str) -> None:
    """TREC run format, one line per (query, document): qid Q0 doc_id rank score tag"""
    with open(path, "w", encoding="utf-8") as f:
        for qid, scores in run.items():
            for rank, doc_id in enumerate(ranking(scores), start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {scores[doc_id]:.6f} {tag}\n")


def read_run(path: str) -> dict[str, dict[str, float]]:
    """Inverse of write_run: {qid: {doc_id: score}}. IDs stay strings, scores become floats."""
    run: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, _, score, _ = line.split()
            if qid not in run:
                run[qid] = {}
            run[qid][doc_id] = float(score)
    return run


def per_query_metrics(run, qrels) -> dict[str, dict[str, float]]:
    """{qid: {metric: value}} for every query in qrels; missing queries score 0."""
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


def mean_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average each metric over all queries."""
    n = len(per_query)
    return {m: sum(q[m] for q in per_query.values()) / n for m in METRICS}


def pytrec_means(run, qrels) -> dict[str, float]:
    """Reference values from pytrec_eval, averaged the same way (all qrels queries, missing = 0).
    Judged@10 has no pytrec_eval equivalent, so it is not included."""
    nonempty = {q: docs for q, docs in run.items() if docs}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "ndcg_cut.5,10", "P.10", "recall.100"})
    res = evaluator.evaluate(nonempty)
    names = {"NDCG@5": "ndcg_cut_5", "NDCG@10": "ndcg_cut_10", "MAP": "map",
             "P@10": "P_10", "R@100": "recall_100"}
    return {mine: sum(res.get(q, {}).get(theirs, 0.0) for q in qrels) / len(qrels)
            for mine, theirs in names.items()}
