import json
from collections import defaultdict
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

if __name__ == "__main__":
    corpus = load_corpus(r"data\nfcorpus\corpus.jsonl")
    print(len(corpus))
    nfcorpus = load_corpus(r"data\nfcorpus\corpus.jsonl")
    print(len(nfcorpus))