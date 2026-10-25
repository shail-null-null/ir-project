from src.data import load_corpus, load_queries, load_qrels, filter_queries

for name in ["scifact", "nfcorpus"]:
    base = f"data/{name}"
    corpus = load_corpus(f"{base}/corpus.jsonl")
    qrels = load_qrels(f"{base}/qrels/test.tsv")
    queries = filter_queries(load_queries(f"{base}/queries.jsonl"), qrels)
    print(f"\n===== {name} =====")
    for qid in list(qrels)[:5]:
        best = max(qrels[qid], key=qrels[qid].get)   # highest-graded relevant doc
        doc = corpus[best]
        print(f"\nQUERY: {queries[qid]}")
        print(f"REL DOC: {doc['title']} | {doc['text'][:300]}")