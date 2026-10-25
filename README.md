# Domain-Specific Search: BM25 vs. Dense Retrieval on Technical Documents

Information Retrieval course project — **Group 8**
Shail Kamlesh Shah · Milap Dineshkumar Chaudhari · Vaibhav Bhatnagar

## Overview
A comparative study of classical lexical retrieval (BM25) and dense neural retrieval on
domain-specific corpora (biomedical, computer science, legal). Beyond overall scores, the
project analyses *where* and *why* each method wins — per domain and per query type.

## Status
- [x] BEIR-format data loaders with integrity checks (SciFact, NFCorpus)
- [x] Evaluation module: P@k, AP/MAP, NDCG@k — unit-tested and validated against `pytrec_eval`
- [ ] BM25 retrieval, implemented from scratch — in progress
- [ ] Dense retrieval (Qwen3-Embedding) — planned
- [ ] Additional domains: CS (SciDocs, CQADupStack), legal (AILA) — planned
- [ ] Query-category analysis — planned

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
```bash
python src/data.py       # load datasets and run integrity checks
python -m pytest -v      # run evaluator tests
```

## Repository structure
```
src/
  data.py          loaders for corpus / queries / qrels + integrity checks
  metrics.py       P@k, AP, NDCG@k and per-query evaluation
tests/
  test_metrics.py  unit tests incl. edge cases + cross-check against pytrec_eval
runs/              retrieval runs in TREC format (not committed)
```

## Evaluation conventions
- A document is relevant if its grade is ≥ 1 (used by P@k and AP).
- NDCG uses linear gain (the grade) with a log2(rank + 1) discount, matching `trec_eval`.
- Metrics are averaged over **all** test queries in the qrels; a query with no retrieved
  documents scores 0.