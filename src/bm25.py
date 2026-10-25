"""BM25 retrieval, implemented from scratch.

Lucene-style BM25. Differs from BEIR's Anserini baseline (k1=0.9, b=0.4, separate
title/body fields, Anserini's analyzer) — see the status report for the measured effect.
Configuration:
  - document text = title + " " + text
  - lowercase, alphanumeric tokens, Lucene's default English stopword list, no stemming
  - k1 = 1.2, b = 0.75 (Lucene / Elasticsearch defaults)
  - Lucene IDF: ln(1 + (N - df + 0.5) / (df + 0.5)), always positive
  - each distinct query term counted once
"""
import heapq
import math
import re
from collections import Counter, defaultdict

# Lucene's default English stopword list (note: contains "no" and "not")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens, drop stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[str, int]]] = {}  # term -> [(doc_id, tf), ...]
        self.idf: dict[str, float] = {}                       # term -> Lucene IDF
        self.doc_len: dict[str, int] = {}                     # doc_id -> number of tokens
        self.norm: dict[str, float] = {}                      # doc_id -> k1 * (1 - b + b * dl / avgdl)
        self.avgdl = 0.0

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        """Build the inverted index from {doc_id: {"title": ..., "text": ...}}."""
        postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for doc_id, doc in corpus.items():
            tokens = tokenize(f"{doc['title']} {doc['text']}")
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term].append((doc_id, tf))
        self.postings = dict(postings)

        n_docs = len(self.doc_len)
        self.avgdl = sum(self.doc_len.values()) / n_docs

        # Lucene IDF: never negative, unlike the classic RSJ form
        self.idf = {
            term: math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

        # The length-normalisation part of the denominator depends only on the
        # document, so compute it once per document instead of once per posting.
        k1, b, avgdl = self.k1, self.b, self.avgdl
        self.norm = {d: k1 * (1 - b + b * dl / avgdl) for d, dl in self.doc_len.items()}

    def score_query(self, query: str) -> dict[str, float]:
        """Term-at-a-time scoring: only documents containing a query term are touched."""
        scores: dict[str, float] = defaultdict(float)
        k1 = self.k1
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if plist is None:          # term never appears in the corpus
                continue
            idf = self.idf[term]
            for doc_id, tf in plist:
                scores[doc_id] += idf * tf * (k1 + 1) / (tf + self.norm[doc_id])
        return scores

    def search(self, queries: dict[str, str], k: int = 1000) -> dict[str, dict[str, float]]:
        """Return a run: {qid: {doc_id: score}} with the top-k documents per query."""
        run: dict[str, dict[str, float]] = {}
        for qid, text in queries.items():
            scores = self.score_query(text)
            run[qid] = dict(heapq.nlargest(k, scores.items(), key=lambda item: item[1]))
        return run