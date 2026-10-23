import json
from collections import defaultdict
def load_corpus(path):
    with open(path) as f:
    # print(f.readline())
        l = f.readlines()
    d = []
    for i in range(len(l)):
        d.append(json.loads(l[i]))
    dic = defaultdict(list)
    # print(type(d[0]))
    for i in range(len(d)):
        dic[d[i]['_id']].append(d[i]['title'])
        dic[d[i]['_id']].append(d[i]['text'])

    return dic

if __name__ == "__main__":
    corpus = load_corpus(r"D:\ir-project\data\scifact\corpus.jsonl")
    print(len(corpus))
    nfcorpus = load_corpus(r"D:\ir-project\data\nfcorpus\corpus.jsonl")
    print(len(nfcorpus))