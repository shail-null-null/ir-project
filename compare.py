"""Query-by-query comparison of BM25 and dense retrieval, and the checks for H1-H4.

Run from the project root, after evaluate_runs.py:   python compare.py
Writes runs/analysis_summary.json, which make_charts.py and the deck use.
"""
import json
import re
from statistics import mean

from src.bm25 import BM25, tokenize
from src.data import filter_queries, load_corpus, load_qrels, load_queries
from src.evaluation import ranking, read_run
from src.significance import paired_randomization_test

DATASETS = ["scifact", "nfcorpus", "scidocs"]
METRIC = "NDCG@10"
ALPHA = 0.05


def load_per_query(system, dataset):
    with open(f"runs/perquery_{system}_{dataset}.json", encoding="utf-8") as f:
        return json.load(f)


def load_queries_for(dataset):
    base = f"data/{dataset}"
    qrels = load_qrels(f"{base}/qrels/test.tsv")
    return filter_queries(load_queries(f"{base}/queries.jsonl"), qrels), qrels


def sig(p):
    return f"p = {p:.4f} ({'significant' if p < ALPHA else 'not significant'} at {ALPHA})"


def compare_subset(qids, bm25, dense):
    """Means, mean difference (dense - BM25) and p-value over a set of queries."""
    a = [dense[q][METRIC] for q in qids]
    b = [bm25[q][METRIC] for q in qids]
    return {"n": len(qids), "bm25": mean(b), "dense": mean(a),
            "diff": mean(a) - mean(b), "p": paired_randomization_test(a, b)}


summary = {}

# ---------------------------------------------------------------------------
# 1. Per dataset: overall difference, wins / losses / ties, biggest wins
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"1. BM25 vs dense, query by query ({METRIC})")
print("=" * 80)
per_query = {}
for dataset in DATASETS:
    bm25, dense = load_per_query("bm25", dataset), load_per_query("dense", dataset)
    per_query[dataset] = (bm25, dense)
    queries, _ = load_queries_for(dataset)
    qids = list(bm25)
    diff = {q: dense[q][METRIC] - bm25[q][METRIC] for q in qids}
    res = compare_subset(qids, bm25, dense)
    wins = sum(1 for d in diff.values() if d > 1e-9)
    losses = sum(1 for d in diff.values() if d < -1e-9)
    ties = len(qids) - wins - losses
    summary[dataset] = {**res, "dense_wins": wins, "bm25_wins": losses, "ties": ties,
                        "diffs": list(diff.values())}

    print(f"\n{dataset}: {len(qids)} queries | BM25 {res['bm25']:.4f} | dense {res['dense']:.4f} | "
          f"dense - BM25 = {res['diff']:+.4f} | {sig(res['p'])}")
    print(f"  dense better on {wins}, BM25 better on {losses}, tied on {ties}")
    ordered = sorted(qids, key=lambda q: diff[q])
    print("  biggest dense wins:")
    for q in [q for q in reversed(ordered) if diff[q] > 1e-9][:3]:
        print(f"    {diff[q]:+.3f}  [{q}] {queries[q][:75]}")
    print("  biggest BM25 wins:")
    for q in [q for q in ordered if diff[q] < -1e-9][:3]:
        print(f"    {diff[q]:+.3f}  [{q}] {queries[q][:75]}")

# ---------------------------------------------------------------------------
# 2. H2 and H4 (overall, per dataset)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. H2 and H4")
print("=" * 80)
sf, nf = summary["scifact"], summary["nfcorpus"]
h2 = nf["diff"] > sf["diff"]
print(f"H2: dense advantage NFCorpus {nf['diff']:+.4f} vs SciFact {sf['diff']:+.4f} -> "
      f"{'SUPPORTED' if h2 else 'REFUTED'} (judged on direction, as pre-registered)")
h4a = sf["bm25"] > sf["dense"]
print(f"H4a (BM25 beats dense on SciFact): {'SUPPORTED' if h4a else 'REFUTED'} | {sig(sf['p'])}")
h4b = nf["dense"] > nf["bm25"]
print(f"H4b (dense beats BM25 on NFCorpus): {'SUPPORTED' if h4b else 'REFUTED'} | {sig(nf['p'])}")
print("SciDocs: no pre-registered hypothesis -> exploratory result only")
summary["H2"] = {"verdict": "supported" if h2 else "refuted"}
summary["H4a"] = {"verdict": "supported" if h4a else "refuted", "p": sf["p"]}
summary["H4b"] = {"verdict": "supported" if h4b else "refuted", "p": nf["p"]}

# ---------------------------------------------------------------------------
# 3. H1: the NFCorpus vocabulary-gap queries
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. H1: NFCorpus queries with no term in the index (not fixable by plurals)")
print("=" * 80)
nf_corpus = load_corpus("data/nfcorpus/corpus.jsonl")
nf_queries, _ = load_queries_for("nfcorpus")
index = BM25()
index.index(nf_corpus)
vocab = index.postings


def variants(t):
    v = {t + "s", t + "es"}
    if t.endswith("ies"):
        v.add(t[:-3] + "y")
    if t.endswith("es"):
        v.add(t[:-2])
    if t.endswith("s"):
        v.add(t[:-1])
    return {x for x in v if x in vocab}


gap_qids = [q for q, text in nf_queries.items()
            if not any(t in vocab for t in tokenize(text))
            and not any(variants(t) for t in tokenize(text))]
print(f"{len(gap_qids)} vocabulary-gap queries (the pre-registered set has 18)")

bm25_nf, dense_nf = per_query["nfcorpus"]
dense_run = read_run("runs/dense_nfcorpus.trec")
above_zero = 0
h1_rows = []
for q in gap_qids:
    score = dense_nf[q][METRIC]
    above_zero += score > 0
    top3 = [nf_corpus[d]["title"][:55] for d in ranking(dense_run.get(q, {}))[:3]]
    h1_rows.append({"qid": q, "query": nf_queries[q], "dense_ndcg10": score, "top3": top3})
    print(f"  {score:.3f}  {nf_queries[q]:<16} | dense top 3: {top3}")
h1 = above_zero == len(gap_qids)
print(f"H1 (dense NDCG@10 > 0 on ALL of them): {above_zero}/{len(gap_qids)} above 0 -> "
      f"{'SUPPORTED' if h1 else 'REFUTED'}")
summary["H1"] = {"verdict": "supported" if h1 else "refuted", "above_zero": above_zero,
                 "n": len(gap_qids), "queries": h1_rows}

# ---------------------------------------------------------------------------
# 4. H3: SciFact queries containing identifiers
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. H3: SciFact queries containing an identifier (a token with a letter AND a digit)")
print("=" * 80)
sf_queries, _ = load_queries_for("scifact")
bm25_sf, dense_sf = per_query["scifact"]


def has_identifier(text):
    tokens = re.findall(r"[A-Za-z0-9-]+", text)   # split on whitespace/punctuation, keep hyphens
    return any(re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t) for t in tokens)


ident = [q for q in sf_queries if has_identifier(sf_queries[q])]
rest = [q for q in sf_queries if q not in set(ident)]
r_ident = compare_subset(ident, bm25_sf, dense_sf)
r_rest = compare_subset(rest, bm25_sf, dense_sf)
print(f"with identifiers   : n={r_ident['n']:3d} | BM25 {r_ident['bm25']:.4f} | dense {r_ident['dense']:.4f} | "
      f"dense - BM25 = {r_ident['diff']:+.4f} | {sig(r_ident['p'])}")
print(f"without identifiers: n={r_rest['n']:3d} | BM25 {r_rest['bm25']:.4f} | dense {r_rest['dense']:.4f} | "
      f"dense - BM25 = {r_rest['diff']:+.4f} | {sig(r_rest['p'])}")
h3 = r_ident["bm25"] > r_ident["dense"]
print(f"H3 (BM25 beats dense on identifier queries): {'SUPPORTED' if h3 else 'REFUTED'}")
print("   examples:", [sf_queries[q][:50] for q in ident[:3]])
summary["H3"] = {"verdict": "supported" if h3 else "refuted", "identifier": r_ident, "rest": r_rest}

# ---------------------------------------------------------------------------
# 5. NFCorpus grade-0 check: do the systems retrieve documents judged NOT relevant?
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. NFCorpus: grade-0 (judged non-relevant) documents in each system's top 10")
print("=" * 80)
_, nf_qrels = load_queries_for("nfcorpus")
grade0 = sum(1 for rels in nf_qrels.values() for g in rels.values() if g == 0)
print(f"grade-0 judgments in NFCorpus test qrels: {grade0}")
for system in ["bm25", "dense"]:
    run = read_run(f"runs/{system}_nfcorpus.trec")
    hits = sum(1 for q, rels in nf_qrels.items()
               for d in ranking(run.get(q, {}))[:10] if rels.get(d) == 0)
    print(f"  {system}: {hits} grade-0 documents in top 10 across all queries")
    summary.setdefault("grade0_top10", {})[system] = hits

with open("runs/analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
print("\nsaved runs/analysis_summary.json")
