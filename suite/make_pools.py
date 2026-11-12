"""Pool the top 10 of BM25 and of dense for each judged query, shuffled and anonymised.

Run from the project root (after both suite runs exist):   python -m suite.make_pools
Writes suite/pools/<query_id>.json: the query, its criterion, and the pooled documents
(id, title, text) in random order, with no system names or ranks.
"""
import json
import os
import random

from src.data import load_corpus
from src.evaluation import ranking, read_run

DEPTH = 10
corpus = load_corpus("data/nfcorpus/corpus.jsonl")
runs = {s: read_run(f"runs/suite_{s}_nfcorpus.trec") for s in ["bm25", "dense"]}
with open("suite/judged_queries.json", encoding="utf-8") as f:
    judged = json.load(f)["queries"]

os.makedirs("suite/pools", exist_ok=True)
rng = random.Random(2026)
for j in judged:
    qid = j["id"]
    pool = set()
    for run in runs.values():
        pool.update(ranking(run.get(qid, {}))[:DEPTH])
    docs = sorted(pool)
    rng.shuffle(docs)
    out = {"query_id": qid, "query": j["text"], "criterion": j["criterion"],
           "documents": [{"doc_id": d, "title": corpus[d]["title"], "text": corpus[d]["text"]}
                         for d in docs]}
    with open(f"suite/pools/{qid}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"{qid}: {len(docs)} pooled documents")
