"""List NFCorpus test queries that match no indexed term, and test the plural/singular explanation."""
from src.bm25 import BM25, TOKEN_RE, tokenize
from src.data import filter_queries, load_corpus, load_qrels, load_queries

base = "data/nfcorpus"
corpus = load_corpus(f"{base}/corpus.jsonl")
qrels = load_qrels(f"{base}/qrels/test.tsv")
queries = filter_queries(load_queries(f"{base}/queries.jsonl"), qrels)

bm25 = BM25()
bm25.index(corpus)
vocab = bm25.postings


def variants(t: str) -> set[str]:
    """Crude plural/singular variants of t that DO exist in the corpus vocabulary."""
    v = {t + "s", t + "es"}
    if t.endswith("ies"):
        v.add(t[:-3] + "y")
    if t.endswith("es"):
        v.add(t[:-2])
    if t.endswith("s"):
        v.add(t[:-1])
    return {x for x in v if x in vocab}


empty = [(qid, text) for qid, text in queries.items()
         if not any(t in vocab for t in tokenize(text))]
print(f"{len(empty)} queries match no indexed term\n")
for qid, text in empty:
    raw = TOKEN_RE.findall(text.lower())
    toks = tokenize(text)
    print(f"{qid} | {text!r}")
    print(f"    tokens={toks}  stopwords dropped={[t for t in raw if t not in toks]}  "
          f"in-vocab variants={ {t: sorted(variants(t)) for t in toks if variants(t)} }")

rescued = sum(1 for _, text in empty if any(variants(t) for t in tokenize(text)))
print(f"\n{rescued}/{len(empty)} would match at least one term if plural/singular were normalised")