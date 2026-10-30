from src.bm25 import tokenize
from src.data import filter_queries, load_corpus, load_qrels, load_queries

for name in ["scifact", "nfcorpus"]:
    base = f"data/{name}"
    corpus = load_corpus(f"{base}/corpus.jsonl")
    qrels = load_qrels(f"{base}/qrels/test.tsv")
    queries = filter_queries(load_queries(f"{base}/queries.jsonl"), qrels)

    overlaps = []
    for qid, rels in qrels.items():
        best = max(rels, key = rels.get)        # TODO 1: highest-graded relevant doc (same as peek.py)
        q = set(tokenize(queries[qid]))           # TODO 2: set of the query's tokens
        d = set(tokenize(corpus[best]["title"] + " " + corpus[best]["text"]))           # TODO 3: set of the doc's tokens (title + " " + text)
        if not q:
            continue
        overlaps.append(len(q&d)/len(q))   # TODO 4: fraction of query terms that appear in the doc

    n = len(overlaps)
    print(f"{name}: {n} queries measured")
    print(f"  mean overlap   = {sum(overlaps) / n:.3f}")
    print(f"  overlap == 0   = {100 * sum(1 for o in overlaps if o == 0) / n:.1f}%")
    print(f"  overlap == 1.0 = {100 * sum(1 for o in overlaps if o == 1.0) / n:.1f}%")