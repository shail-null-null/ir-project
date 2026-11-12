"""Vocabulary check for the context-enrichment descriptors (a fact about the corpus, not a result).

Run from the project root:   python -m suite.check_descriptors
"""
from src.bm25 import BM25, tokenize
from src.data import load_corpus

corpus = load_corpus("data/nfcorpus/corpus.jsonl")
bm25 = BM25()
bm25.index(corpus)

for query in ["Zoloft antidepressant", "Mevacor cholesterol drug", "Splenda artificial sweetener"]:
    print(f"\n{query}")
    for word in tokenize(query):
        for form in sorted({word, word + "s", word[:-1] if word.endswith("s") else word}):
            df = len(bm25.postings.get(form, []))
            print(f"  {form:<16} {'IN vocabulary' if df else 'not in vocabulary':<18} df={df}")
