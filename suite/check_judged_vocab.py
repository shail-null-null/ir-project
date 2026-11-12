"""Check that each judged query's key words exist in NFCorpus (a corpus fact, not a result).

Run from the project root:   python -m suite.check_judged_vocab
"""
from src.bm25 import BM25
from src.data import load_corpus

corpus = load_corpus("data/nfcorpus/corpus.jsonl")
bm25 = BM25()
bm25.index(corpus)

KEY_WORDS = ["cholesterol", "medication", "calcium", "dairy", "plaque", "atherosclerosis",
             "cold", "colds", "infection", "vitamin", "iu", "fasting", "hours", "prolonged"]
for w in KEY_WORDS:
    df = len(bm25.postings.get(w, []))
    print(f"{w:<16} df={df}")
