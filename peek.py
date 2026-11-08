from src.data import load_corpus, load_queries, load_qrels, filter_queries

# for name in ["scifact", "nfcorpus"]:
#     base = f"data/{name}"
#     corpus = load_corpus(f"{base}/corpus.jsonl")
#     qrels = load_qrels(f"{base}/qrels/test.tsv")
#     queries = filter_queries(load_queries(f"{base}/queries.jsonl"), qrels)
#     print(f"\n===== {name} =====")
#     for qid in list(qrels)[:5]:
#         best = max(qrels[qid], key=qrels[qid].get)   # highest-graded relevant doc
#         doc = corpus[best]
#         print(f"\nQUERY: {queries[qid]}")
#         print(f"REL DOC: {doc['title']} | {doc['text'][:300]}")
import random
random.seed(0)


datasets = ["nfcorpus","scifact"]
for dataset in datasets:
    print(f"__________{dataset}___________")
    queries = load_queries(f"data/{dataset}/queries.jsonl")
    test_qrels = load_qrels(f"data/{dataset}/qrels/test.tsv")
    corpus = load_corpus(f"data/{dataset}/corpus.jsonl")
    # count = 0
    for qid in random.sample(list(test_qrels), 10):
        print(queries[qid][:300])
        max_corpus = max(test_qrels[qid],key=test_qrels[qid].get)
        print(corpus[max_corpus]["title"])
        print(corpus[max_corpus]["text"][0:300])
        print("\n")
    print("\n")
    print("\n")