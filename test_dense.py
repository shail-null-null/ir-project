# ---- Colab cell: test DenseRetriever on a 50-document SciFact subset ----
# Run from the repo root (after `%cd ir-project`), with src/dense.py in place.
import random
import torch
from src.data import filter_queries, load_corpus, load_qrels, load_queries
from src.dense import DenseRetriever

INSTRUCTION = "Given a scientific claim, retrieve documents that support or refute the claim"
corpus = load_corpus("data/scifact/corpus.jsonl")
qrels = load_qrels("data/scifact/qrels/test.tsv")
queries = filter_queries(load_queries("data/scifact/queries.jsonl"), qrels)

# Subset: the relevant docs of 3 test queries + random others, 50 docs in total
random.seed(0)
test_qids = random.sample(list(qrels), 3)
relevant = {d for q in test_qids for d in qrels[q]}
others = random.sample([d for d in corpus if d not in relevant], 50 - len(relevant))
subset = {d: corpus[d] for d in list(relevant) + others}

r = DenseRetriever(instruction=INSTRUCTION, batch_size=8)
print("device:", r.device, "| dtype:", r.dtype)

# Check 1: batching must not change the vectors
two = ["Statins lower cholesterol.",
       "Statins are a class of drugs that lower LDL cholesterol by inhibiting HMG-CoA reductase."]
r.batch_size = 1; a = r.embed(two)
r.batch_size = 2; b = r.embed(two)
r.batch_size = 8
print("1. batching consistent:", torch.allclose(a, b, atol=1e-4), "| shape:", tuple(a.shape))

# Check 2: index shape and norms
r.index(subset)
norms = r.doc_matrix.norm(dim=1)
print("2. doc_matrix:", tuple(r.doc_matrix.shape),
      "| norms in [0.999, 1.001]:", bool(((norms > 0.999) & (norms < 1.001)).all()))

# Check 3: run format
k = 10
run = r.search({q: queries[q] for q in test_qids}, k=k)
ok_len = all(len(run[q]) == min(k, len(subset)) for q in test_qids)
ok_ids = all(d in subset for q in test_qids for d in run[q])
ok_sorted = all(list(run[q].values()) == sorted(run[q].values(), reverse=True) for q in test_qids)
print(f"3. {k} results per query: {ok_len} | IDs are real doc IDs: {ok_ids} | scores sorted: {ok_sorted}")

# Check 4: alignment -- recompute one score by hand and compare with the run
q0 = test_qids[0]
top_doc, top_score = next(iter(run[q0].items()))
q_vec = r.embed([f"Instruct: {INSTRUCTION}\nQuery:{queries[q0]}"])[0]
manual = float(q_vec @ r.doc_matrix[r.doc_ids.index(top_doc)])
print(f"4. alignment: run score {top_score:.5f} vs recomputed {manual:.5f} ->",
      abs(top_score - manual) < 1e-4)

# Where did each query's relevant document(s) land?
for q in test_qids:
    ranked = list(run[q])
    ranks = [ranked.index(d) + 1 if d in ranked else f">{k}" for d in qrels[q]]
    print(f"   query {q}: relevant doc rank(s) {ranks} | claim: {queries[q][:70]}")
