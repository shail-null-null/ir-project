import json
import pandas as pd
def load_corpus(path):
    with open(path,encoding="utf-8") as f:
    # print(f.readline())
        l = f.readlines()
    d = []
    for i in range(len(l)):
        d.append(json.loads(l[i]))
    dic = {}
    # print(type(d[0]))
    for i in range(len(d)):
        dic[d[i]['_id']]={"title":d[i]["title"],"text":d[i]["text"]}
    return dic

def load_queries(path):
    with open(path,encoding="utf-8") as f:
        l = f.readlines()
    
    d = []
    for i in range(len(l)):
        d.append(json.loads(l[i]))
    dic = {}
    for i in range(len(d)):
        dic[d[i]['_id']] = d[i]['text']
    return dic

# def load_qrels(path):
#     df = pd.read_csv(path,sep='\t')
#     dic = {}
#     for i in range(df["query-id"].count()):
#         if str(df["query-id"][i]) not in dic:
#             dic[str(df["query-id"][i])] = {str(df["corpus-id"][i]):df["score"][i]}
#             continue
#         dic[str(df["query-id"][i])][str(df["corpus-id"][i])] = df["score"][i]
#     return dic

def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            qid, doc_id, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = int(score)
    return qrels

def filter_queries(queries,qrels):
    dic = {}
    for q_id in qrels:
        dic[q_id] = queries[q_id]
    return dic


if __name__ == "__main__":
    scifact_corpus = load_corpus(r"data/scifact/corpus.jsonl")
    print(len(scifact_corpus))
    nfcorpus = load_corpus(r"data/nfcorpus/corpus.jsonl")
    print(len(nfcorpus))
    scifact_queries = load_queries(r"data/scifact/queries.jsonl")
    print(len(scifact_queries))
    nfcorpus_queries = load_queries(r"data/nfcorpus/queries.jsonl")
    print(len(nfcorpus_queries))
    scifact_qrels = load_qrels(r"data/scifact/qrels/test.tsv")
    print(len(scifact_qrels))
    nfcorpus_qrels = load_qrels(r"data/nfcorpus/qrels/test.tsv")
    print(len(nfcorpus_qrels))
    filtered_scifact_queries = filter_queries(scifact_queries,scifact_qrels)
    print(len(filtered_scifact_queries))
    filtered_nfcorpus_queries = filter_queries(nfcorpus_queries,nfcorpus_qrels)
    print(len(filtered_nfcorpus_queries))