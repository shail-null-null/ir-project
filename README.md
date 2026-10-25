# Domain-Specific Search: BM25 vs. Dense Retrieval on Technical Documents

Information Retrieval course project — **Group 8**
Shail Kamlesh Shah · Milap Dineshkumar Chaudhari · Vaibhav Bhatnagar

## Overview
A comparative study of classical lexical retrieval (BM25) and dense neural retrieval on
domain-specific corpora (biomedical, computer science, legal). Beyond overall scores, the
project analyses *where* and *why* each method wins — per domain and per query type.

## Status
- [x] BEIR-format data loaders with integrity checks (SciFact, NFCorpus)
- [x] Evaluation module: P@k, AP/MAP, NDCG@k — 14 unit tests, validated against `pytrec_eval`
- [x] BM25 retrieval, implemented from scratch (inverted index, Lucene IDF)
- [x] Experiment runner: TREC run files, timing, memory, automatic cross-check against `pytrec_eval`
- [ ] Dense retrieval (Qwen3-Embedding-0.6B) — in progress
- [ ] BM25 with stemming and separate title / body fields
- [ ] Additional domains: CS (SciDocs, CQADupStack), legal (AILA)
- [ ] Query-category analysis and significance tests

## Current results (BM25, k1 = 1.2, b = 0.75)

| Dataset  | NDCG@5 | NDCG@10 | MAP    | P@10   | R@100  | Judged@10 |
|----------|--------|---------|--------|--------|--------|-----------|
| SciFact  | 0.6447 | 0.6645  | 0.6258 | 0.0873 | 0.8859 | 0.0873    |
| NFCorpus | 0.3357 | 0.3079  | 0.1429 | 0.2167 | 0.2360 | 0.2829    |

For reference, BEIR's published BM25 reports NDCG@10 of 0.665 (SciFact) and 0.325 (NFCorpus).
25 of 323 NFCorpus queries share no term with the index; 7 of them would match after stemming.

## Setup
Python 3.10+.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Data
Download SciFact and NFCorpus in BEIR format (links in the
[BEIR repository](https://github.com/beir-cellar/beir)) and unzip into `data/`:
```
data/
  scifact/    corpus.jsonl  queries.jsonl  qrels/test.tsv
  nfcorpus/   corpus.jsonl  queries.jsonl  qrels/test.tsv
```
Data files are not committed to the repository.

## Usage
Run everything from the project root.
```bash
python src/data.py          # load datasets and run integrity checks
python -m pytest -v         # run the evaluator test suite
python -m src.run_bm25      # index, retrieve, evaluate, cross-check, print results table
python inspect_empty.py     # list NFCorpus queries that share no term with the index
```
`python -m` is required for the last-but-one command so that `src` is importable.

## Repository structure
```
src/
  data.py          loaders for corpus / queries / qrels + integrity checks
  metrics.py       P@k, AP, NDCG@k and per-query evaluation
  bm25.py          tokenizer and BM25 retriever (inverted index, term-at-a-time scoring)
  run_bm25.py      experiment runner; also defines Recall@k and Judged@k
tests/
  test_metrics.py  unit tests incl. edge cases + cross-check against pytrec_eval
inspect_empty.py   analysis of queries with zero term overlap
runs/              retrieval runs in TREC format (generated, not committed)
```

## Pipeline decisions
- Document text = title + " " + text.
- Tokenizer: lowercase, `[a-z0-9]+` tokens (hyphens split words; non-ASCII characters dropped).
- Stopwords: Lucene's default 33-word English list (includes "no" and "not"). No stemming yet.
- BM25: k1 = 1.2, b = 0.75, Lucene IDF `ln(1 + (N - df + 0.5) / (df + 0.5))`.
- Each distinct query term counted once; top 1000 documents retrieved per query.
- Ties broken by document ID, descending (trec_eval's convention).

## Evaluation conventions
- A document is relevant if its grade is ≥ 1 (used by P@k, R@k and AP).
- NDCG uses linear gain (the grade) with a log2(rank + 1) discount, matching `trec_eval`.
- Metrics are averaged over **all** test queries in the qrels; a query with no retrieved
  documents scores 0 (`pytrec_eval` would silently drop it).
- Judged@10 = fraction of the top 10 that has any relevance judgment.