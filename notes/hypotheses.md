# Hypotheses: BM25 vs. Dense Retrieval

Written and committed **before any dense retrieval run**; the git commit timestamp is the
record. Each hypothesis is a prediction the results can prove wrong.

---

## Observations (from the data, before any dense run)

### O1. Lexical overlap between queries and their best document (`overlap.py`)
Overlap = fraction of a query's distinct tokens (BM25 tokenizer) that appear in its
highest-graded relevant document.

| Dataset  | Queries | Mean overlap | Overlap = 0 | Overlap = 1.0 |
|----------|---------|--------------|-------------|---------------|
| SciFact  | 300     | 0.505        | 2.0%        | 2.0%          |
| NFCorpus | 323     | 0.253        | 55.1%       | 10.8%         |

Caveat: measured against ONE relevant document per query. NFCorpus has ~38 relevant
documents per query, so "55% share no term with their highest-graded relevant document" —
not "with their relevant documents".

### O2. Queries with no term anywhere in the index (`inspect_empty.py`)
25 of 323 NFCorpus queries (7.7%); BM25 returns nothing for them.
- **7 word-form mismatches** (stemming would recover them): scanners, antinutrients, bagels,
  sores, leeks, pineapples, turnips
- **18 true vocabulary gaps** (the term never occurs in the corpus):
  - brand names: Alli, Fosamax, Mevacor, Splenda, Zoloft
  - foods: eggnog, halibut, mesquite, okra, taro, whiting
  - places / institutions: Czechoslovakia, Peoria, Tufts, Yale
  - medical terms: amnesia, deafness, myelopathy

### O3. Who writes the queries
- **SciFact:** expert-written claims, about one sentence (~12 words), reusing the papers'
  own terminology.
- **NFCorpus:** lay queries and article titles from a consumer nutrition website
  (~3 words, ranging 1–10); the documents are PubMed abstracts.

### O4. Exact identifiers (SciFact sample of 10 queries)
mTORC2, xCT, rs647161, DNA, G-to-A, clpC, CX3CR1, Th2, ADHD, co-IR, SHP-2, MAPK.
The BM25 tokenizer fragments some of them: co-IR → `co`, `ir`; SHP-2 → `shp`, `2`;
G-to-A → `g` ("to" and "a" are stopwords).

### O5. Negation
- **On SciFact, negation does not change relevance:** the same paper supports one version of
  a claim and refutes the other. Negated SciFact claims (with inherited labels) therefore
  test *robustness*, not *understanding* of negation.
- **Testing understanding** needs queries where negation changes what is relevant
  (e.g. "diets without dairy"), which require manual relevance judgments.
- **Lexical markers mislead:** "knockin mouse lacking the SHP-2 MAPK pathway" contains
  "lacking" but is not a negated claim, so a rule like "contains not/lack" over-counts.

---

## Hypotheses

**Conventions for all hypotheses**
- Dense system: Qwen3-Embedding-0.6B, exact search. BM25 system: the submitted
  configuration (k1 = 1.2, b = 0.75, no stemming).
- "X beats Y" = higher mean NDCG@10 over the stated queries.
- Where the number of queries allows, differences are also tested for significance with a
  paired randomization test (reported, but the prediction is judged on the direction).

### H1. Vocabulary-gap queries
- **Prediction:** On **all 18** NFCorpus vocabulary-gap queries (O2), dense retrieval scores
  NDCG@10 > 0, where BM25 scores exactly 0.
- **Because:** dense retrieval does not need shared words. It places texts by meaning, so if
  training exposed it to the relationship (e.g. Zoloft ↔ sertraline), a query can reach
  documents that never contain the query word.
- **Queries identified by:** NFCorpus test queries with no token in the BM25 index that are
  not recovered by plural/singular normalisation (the 18 listed in O2).
- **Proven wrong if:** dense NDCG@10 = 0 on **at least one** of the 18 queries.

### H2. Dense retrieval's advantage is larger on NFCorpus than on SciFact
- **Prediction:** (dense − BM25) NDCG@10 is larger on NFCorpus than on SciFact.
- **Because:** dense retrieval does not depend only on exact word matches and can relate
  different words with similar meaning. NFCorpus has much lower query–document overlap
  (0.25 vs. 0.50, O1), so more of its relevant documents are reachable only through meaning.
- **Queries identified by:** all test queries of each dataset.
- **Proven wrong if:** (dense − BM25) NDCG@10 on NFCorpus ≤ (dense − BM25) NDCG@10 on SciFact.

### H3. Queries containing identifiers
- **Prediction:** On SciFact queries containing identifiers, **BM25 beats dense**.
- **Because:** BM25 matches an identifier such as `rs647161` as one exact token, and a rare
  exact token gets a high IDF. A dense model's subword tokenizer splits such identifiers into
  pieces, so the exact match is diluted.
  *Verified with the Qwen3 tokenizer: identifiers split into many pieces, every digit
  separately (`rs647161` → `rs` + 6 single digits; `CX3CR1` → `CX`, `3`, `CR`, `1`;
  `Fosamax` → `F`, `os`, `amax`; `mTORC2` → `m`, `TOR`, `C`, `2`). Identifiers that differ by
  one character (`rs647161` vs. `rs647162`) share almost all their pieces, so dense retrieval
  cannot reliably tell them apart, while BM25 treats them as different tokens. The reverse
  effect is also possible: dense retrieval may match an identifier written differently in a
  document (e.g. "CX3C chemokine receptor 1"); the analysis will look for both.*
- **Queries identified by:** SciFact test queries containing at least one token with both a
  letter and a digit (e.g. rs647161, CX3CR1, Th2, mTORC2), splitting the original query text
  on whitespace and punctuation other than hyphens.
- **Proven wrong if:** dense mean NDCG@10 ≥ BM25 mean NDCG@10 on these queries.

### H4. Does BEIR's finding still hold for a 2025 embedding model?
- **Prediction:**
  - **H4a:** On **SciFact**, BM25 beats Qwen3 (overall NDCG@10).
  - **H4b:** On **NFCorpus**, Qwen3 beats BM25 (overall NDCG@10).
- **Because:** query–document overlap, not model size or training scale, decides the winner.
  SciFact's queries and corpus are technical and share exact terminology, which favours
  exact matching; NFCorpus's queries are in lay language, which favours matching by meaning.
- **Queries identified by:** all test queries of each dataset.
- **Proven wrong if:** H4a fails if Qwen3 NDCG@10 ≥ BM25 on SciFact; H4b fails if Qwen3
  NDCG@10 ≤ BM25 on NFCorpus. The two parts are judged separately.