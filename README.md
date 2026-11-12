# Domain-Specific Search: BM25 vs. Dense Retrieval

Information Retrieval project,
Shail Kamlesh Shah 

## Overview

We compare a lexical retriever (BM25, implemented from scratch) with a dense
retriever (Qwen3-Embedding-0.6B, zero-shot) on three BEIR collections: SciFact,
NFCorpus and SciDocs. Beyond overall scores, the project asks *when* each method
wins and *why*, using a stress-test suite of 23 queries and pre-registered
hypotheses committed before the dense runs.

## Status

- [x] BEIR-format data loaders with integrity checks (SciFact, NFCorpus, SciDocs)
- [x] Evaluator: P@k, R@k, AP/MAP, NDCG@k, Judged@k; 28 unit tests; agrees with `pytrec_eval` to ~1e-16
- [x] BM25 retrieval, implemented from scratch (inverted index, Lucene IDF)
- [x] Dense retrieval with Qwen3-Embedding-0.6B (exact cosine search; Colab T4 GPU)
- [x] Significance: paired randomisation test (10,000 sign-flips)
- [x] Hypotheses H1–H4 and H5 (`notes/hypotheses.md`), written before the dense runs
- [x] Stress-test suite: 17 inherited-label variants and 6 judged queries (`suite/`)
- [ ] BM25 with stemming and separate title / body fields (planned ablation)
- [ ] Reciprocal Rank Fusion of existing runs
- [ ] Re-ranking, larger embedding models, domain-specific models (MedCPT, SPECTER2)
- [ ] Legal domain (AILA) and CQADupStack

## Results

NDCG@10 on the test splits. BM25 uses k1 = 1.2, b = 0.75; dense is Qwen3-Embedding-0.6B.

| Dataset  | System | NDCG@10 | MAP    | R@100  | Dense − BM25 | p (randomisation) |
|----------|--------|---------|--------|--------|--------------|-------------------|
| SciFact  | BM25   | 0.6645  | 0.6258 | 0.8859 | +0.036       | 0.066 (n.s.)      |
| SciFact  | dense  | 0.7005  | 0.6569 | 0.9633 |              |                   |
| NFCorpus | BM25   | 0.3079  | 0.1429 | 0.2360 | +0.054       | < 0.0001          |
| NFCorpus | dense  | 0.3618  | 0.1838 | 0.3272 |              |                   |
| SciDocs  | BM25   | 0.1539  | 0.1071 | 0.3497 | +0.089       | < 0.0001 (exploratory) |
| SciDocs  | dense  | 0.2433  | 0.1768 | 0.5300 |              |                   |

Reference: BEIR's published BM25 scores are 0.665 (SciFact), 0.325 (NFCorpus) and
0.158 (SciDocs), using Anserini with k1 = 0.9, b = 0.4 and (per BEIR's documentation)
separate title and body fields. Our BM25 with BEIR's k1 and b scores 0.6608 (SciFact)
and 0.3065 (NFCorpus), so the parameters do not explain the NFCorpus gap; the likely
cause is text processing, which is not yet isolated.

Full per-query results and the hypothesis verdicts are in `runs/` and `notes/hypotheses.md`
(generated, see Reproduce below).

## Data

Download SciFact, NFCorpus and SciDocs in BEIR format from the
[BEIR repository](https://github.com/beir-cellar/beir) and unzip into `data/`:

```
data/
  scifact/   corpus.jsonl  queries.jsonl  qrels/test.tsv
  nfcorpus/  corpus.jsonl  queries.jsonl  qrels/test.tsv
  scidocs/   corpus.jsonl  queries.jsonl  qrels/test.tsv
```

Data files are not committed. NFCorpus relevance judgments come from citation links on
NutritionFacts.org (a direct citation is grade 2, an indirect one grade 1).

## Setup

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Dense retrieval needs PyTorch and Hugging Face `transformers`. It was run on a Colab T4 GPU
(float16 on GPU, float32 on CPU). BM25 runs on a CPU laptop.

## Reproduce

Run everything from the project root. Commands are listed in the order used in the study.

```bash
python -m pytest -v                 # evaluator and significance tests
python -m src.run_bm25              # BM25: index, retrieve, evaluate, cross-check against pytrec_eval
python inspect_empty.py             # NFCorpus queries with no indexed term
python overlap.py                   # query-document word overlap (observations O1)
# dense: run the Colab cell in colab_dense_runs.py (saved from the Colab session), then copy its output to runs/
python evaluate_runs.py             # per-query metrics and results table
python compare.py                   # dense vs BM25 differences and significance
python make_charts.py               # figures/
python -m suite.build_suite         # stress-test suite files
python -m suite.make_pools          # pooled documents for judging (top 10 of each system)
python -m suite.evaluate_suite      # merge judgments, evaluate, check predictions
```

## Repository structure

```
src/
  data.py            loaders for corpus, queries and qrels; integrity checks
  metrics.py         P@k, R@k, AP, NDCG@k, Judged@k per query
  evaluation.py      TREC run files (read/write), ranking with trec_eval tie-breaking, pytrec_eval cross-check
  significance.py    paired randomisation test
  bm25.py            tokeniser and BM25 retriever (inverted index, term-at-a-time scoring)
  run_bm25.py        BM25 experiment runner
  dense.py           Qwen3-Embedding dense retriever (last-token pooling, exact search)
tests/
  test_metrics.py       22 tests, including adversarial cases
  test_evaluation.py    3 tests
  test_significance.py  3 tests, including a scipy cross-check
notes/
  hypotheses.md      observations O1–O5 and hypotheses H1–H5, with falsification criteria
root scripts
  evaluate_runs.py  compare.py  make_charts.py  check_dense.py  peek.py  overlap.py  inspect_empty.py
suite/
  select_sources.py     seeded selection of inherited-label sources
  sources.json          source queries and their variants
  judged_queries.json   six judged queries, criteria and pre-registered predictions
  build_suite.py        builds suite queries and inherited qrels
  run_suite_bm25.py     BM25 runs on the suite
  colab_suite_dense.py  dense runs on the suite (Colab)
  make_pools.py         pools top 10 of each system, shuffled, for judging
  evaluate_suite.py     merges judgments, evaluates, checks predictions
  check_descriptors.py  vocabulary check for context-enrichment descriptors
  check_judged_vocab.py vocabulary check for judged-query terms
  judgments/            six judgment files (.tsv) and flags.json
  pools/                pooled documents per judged query
figures/
  results_ndcg10.png  perquery_diffs.png  suite_changes.png
```

Not committed: `data/`, `runs/`, `.venv/`.

## Method notes

- **Document text:** title + " " + text, for both BM25 and dense.
- **BM25 tokeniser:** lowercase, `[a-z0-9]+` tokens (hyphens split words; non-ASCII characters are dropped).
- **Stopwords:** Lucene's default 33-word English list, which includes "no" and "not". No stemming.
- **BM25:** k1 = 1.2, b = 0.75, Lucene IDF `ln(1 + (N − df + 0.5) / (df + 0.5))`. Each distinct query term counts once.
- **Dense:** Qwen3-Embedding-0.6B, last-token pooling read from the attention mask (the pad token equals end-of-text), L2 normalisation, 512-token truncation. Queries get a task instruction; documents do not.
- **Runs:** top 1000 documents per query, TREC format. Ties broken by document ID, descending (trec_eval's convention).

## Evaluation conventions

- A document is relevant if its grade is ≥ 1 (used by P@k, R@k and AP). NFCorpus has grades 1 and 2; SciFact and SciDocs are binary.
- NDCG uses linear gain with a log2(rank + 1) discount, matching `trec_eval`.
- Means run over **all** test queries in the qrels. A query with no retrieved documents scores 0. `pytrec_eval` would silently drop it, so we average it in ourselves.
- Judged@k = fraction of the top k that has any relevance judgment.
- Significance: paired randomisation test on per-query differences, 10,000 sign-flips, two-sided. The smallest attainable p is about 0.0001, so such values are reported as p < 0.0001.

## Stress-test suite and judgments

- **Inherited-label variants (17):** rewrites of existing queries that keep the original judgments. The 14 source queries (identifier, negation and lay rewrites) were sampled by rule with seed 2026; the technical and context-enrichment variants are written by hand. Context enrichment was added after H1 failed and is labelled error-driven.
- **Judged queries (6):** new queries whose relevance changes with the phrasing (negation, homonyms, a numeric constraint and a duration). Criteria were written before any system was run.
- **Judging:** pooled from the top 10 of BM25 and dense, shuffled, with no system names or ranks, and graded 0/1/2 against the written criteria.
- **Limits:** one judge, no agreement measured, n = 2–6 per category, and scores on judged queries are relative to the pool.

## Limitations

- Unjudged documents count as non-relevant, which may understate dense retrieval.
- NFCorpus relevance is defined by citation links, so an on-topic paper that was never cited counts as a miss.
- One embedding model, used zero-shot with one fixed instruction per dataset. BM25 was run at its default settings and not tuned.
- The cause of the NFCorpus gap to BEIR's BM25 is not isolated.
- No correction for multiple comparisons was applied. The SciFact result (p = 0.066) would not survive one, and the NFCorpus and SciDocs results are far below any adjusted threshold.