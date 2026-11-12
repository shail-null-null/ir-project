"""Run BM25 on every suite query. Run from the project root:   python -m suite.run_suite_bm25
Writes runs/suite_bm25_<dataset>.trec
"""
from src.bm25 import BM25
from src.data import load_corpus, load_queries
from src.evaluation import write_run

for ds in ["scifact", "nfcorpus"]:
    corpus = load_corpus(f"data/{ds}/corpus.jsonl")
    queries = load_queries(f"suite/{ds}/queries.jsonl")
    bm25 = BM25()
    bm25.index(corpus)
    run = bm25.search(queries, k=1000)
    write_run(run, f"runs/suite_bm25_{ds}.trec", "bm25")
    empty = [q for q in queries if not run.get(q)]
    print(f"{ds}: {len(queries)} queries, no results for: {empty or 'none'}")
