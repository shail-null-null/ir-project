"""Evaluate the stress-test suite (23 queries) for BM25 and dense retrieval.

Run from the project root, after the judgments are saved:   python -m suite.evaluate_suite

Needs: suite/metadata.json, suite/<dataset>/qrels/test.tsv (inherited labels),
       suite/judgments/*.tsv (judged labels) and suite/judgments/flags.json,
       suite/pools/*.json, runs/suite_{bm25,dense}_{scifact,nfcorpus}.trec,
       runs/perquery_{bm25,dense}_{scifact,nfcorpus}.json (written by evaluate_runs.py).
Writes: suite/nfcorpus/qrels/full.tsv, runs/suite_summary.json, figures/suite_changes.png
"""
import json
import os
from statistics import mean

from src.data import load_corpus, load_qrels
from src.evaluation import per_query_metrics, ranking, read_run

SYSTEMS = ["bm25", "dense"]
DATASETS = ["scifact", "nfcorpus"]
METRIC = "NDCG@10"
EPS = 1e-9
CATEGORIES = ["lay_rewrite", "negation_flip", "identifier_only", "technical_rewrite", "context_enrichment"]


def short(text, n=60):
    return text if len(text) <= n else text[:n - 1] + "…"


with open("suite/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
with open("suite/judgments/flags.json", encoding="utf-8") as f:
    flags = json.load(f)
judged_ids = [q for q, m in meta.items() if m["judged"]]
inherited_ids = [q for q, m in meta.items() if not m["judged"]]

# ---------------------------------------------------------------------------
# 0. Merge inherited labels with the judged labels, and check the judgments are complete
# ---------------------------------------------------------------------------
qrels = {ds: load_qrels(f"suite/{ds}/qrels/test.tsv") for ds in DATASETS}
for name in sorted(os.listdir("suite/judgments")):
    if name.endswith(".tsv"):
        for qid, rels in load_qrels(f"suite/judgments/{name}").items():
            qrels["nfcorpus"].setdefault(qid, {}).update(rels)

for qid in judged_ids:
    if qid not in qrels["nfcorpus"]:
        raise SystemExit(f"no judgments for {qid} (expected suite/judgments/{qid}.tsv)")
    with open(f"suite/pools/{qid}.json", encoding="utf-8") as f:
        pool = [d["doc_id"] for d in json.load(f)["documents"]]
    missing = [d for d in pool if d not in qrels["nfcorpus"][qid]]
    if missing:
        raise SystemExit(f"{qid}: pooled documents without a grade: {missing}")

with open("suite/nfcorpus/qrels/full.tsv", "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for qid, rels in qrels["nfcorpus"].items():
        for doc_id, grade in rels.items():
            f.write(f"{qid}\t{doc_id}\t{grade}\n")

# ---------------------------------------------------------------------------
# 1. Evaluate every suite query for both systems
# ---------------------------------------------------------------------------
runs = {s: {} for s in SYSTEMS}
results = {s: {} for s in SYSTEMS}
for s in SYSTEMS:
    for ds in DATASETS:
        run = read_run(f"runs/suite_{s}_{ds}.trec")
        runs[s].update(run)
        results[s].update(per_query_metrics(run, qrels[ds]))

# Sanity check: the pools were built from these very runs, so every judged query's top 10
# must be fully judged (Judged@10 = 1.0 for any system that returned results).
for s in SYSTEMS:
    for qid in judged_ids:
        if runs[s].get(qid) and abs(results[s][qid]["Judged@10"] - 1.0) > EPS:
            raise SystemExit(f"{s} {qid}: Judged@10 = {results[s][qid]['Judged@10']:.3f}, expected 1.0. "
                             f"The pools were built from different runs than the ones evaluated here.")
print("check passed: every judged query's top 10 is fully judged for both systems\n")

# ---------------------------------------------------------------------------
# 2. Inherited-label queries: the score change from the original query to its variant
# ---------------------------------------------------------------------------
pq_cache = {}


def source_ndcg(system, qid):
    m = meta[qid]
    key = (system, m["dataset"])
    if key not in pq_cache:
        with open(f"runs/perquery_{system}_{m['dataset']}.json", encoding="utf-8") as f:
            pq_cache[key] = json.load(f)
    return pq_cache[key][m["source_qid"]][METRIC]


rows = []
for qid in inherited_ids:
    m = meta[qid]
    row = {"qid": qid, "category": m["category"], "dataset": m["dataset"],
           "source_text": m["source_text"], "variant": m["text"], "error_driven": m["error_driven"]}
    for s in SYSTEMS:
        row[f"{s}_source"] = source_ndcg(s, qid)
        row[f"{s}_variant"] = results[s][qid][METRIC]
        row[f"{s}_change"] = row[f"{s}_variant"] - row[f"{s}_source"]
    rows.append(row)

print("=" * 96)
print(f"1. Inherited-label queries: {METRIC} of the original query vs its variant (change = variant - original)")
print("=" * 96)
cat_summary = {}
for c in CATEGORIES:
    rs = [r for r in rows if r["category"] == c]
    if not rs:
        continue
    cat_summary[c] = {"n": len(rs), "error_driven": any(r["error_driven"] for r in rs)}
    print(f"\n{c}{'  [ERROR-DRIVEN]' if cat_summary[c]['error_driven'] else ''}  (n = {len(rs)})")
    for r in rs:
        print(f"  {r['qid']:<9} {short(r['variant'], 40):<41} "
              f"BM25 {r['bm25_source']:.3f} -> {r['bm25_variant']:.3f} ({r['bm25_change']:+.3f}) | "
              f"dense {r['dense_source']:.3f} -> {r['dense_variant']:.3f} ({r['dense_change']:+.3f})")
    for s in SYSTEMS:
        cat_summary[c][s] = {
            "source": mean(r[f"{s}_source"] for r in rs),
            "variant": mean(r[f"{s}_variant"] for r in rs),
            "change": mean(r[f"{s}_change"] for r in rs),
            "drops": sum(1 for r in rs if r[f"{s}_change"] < -EPS),
            "gains": sum(1 for r in rs if r[f"{s}_change"] > EPS),
        }
    print("  mean change:  " + " | ".join(
        f"{s} {cat_summary[c][s]['change']:+.3f} (down on {cat_summary[c][s]['drops']}, up on {cat_summary[c][s]['gains']})"
        for s in SYSTEMS))
print("\nNote: with 2-5 queries per category these are per-query observations, not significance tests.")

# ---------------------------------------------------------------------------
# 3. Judged queries: per-query results and the pre-registered predictions
# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("2. Judged queries (NFCorpus; pooled top 10 of both systems, judged blind)")
print("=" * 96)
judged_summary = {}
for qid in judged_ids:
    rels = qrels["nfcorpus"][qid]
    flagged = set(flags.get(qid, {}).get("docs", []))
    entry = {"text": meta[qid]["text"], "category": meta[qid]["category"], "pool_size": len(rels),
             "pool_grades": {g: sum(1 for v in rels.values() if v == g) for g in (0, 1, 2)}}
    print(f"\n{qid}  \"{entry['text']}\"  | pool: {entry['pool_size']} docs, grades 2/1/0 = "
          f"{entry['pool_grades'][2]}/{entry['pool_grades'][1]}/{entry['pool_grades'][0]}")
    for s in SYSTEMS:
        top10 = ranking(runs[s].get(qid, {}))[:10]
        grades = [rels.get(d) for d in top10]
        entry[s] = {"ndcg10": results[s][qid][METRIC], "map": results[s][qid]["MAP"],
                    "returned": len(top10), "grade2": grades.count(2), "grade1": grades.count(1),
                    "grade0": grades.count(0), "flagged": len(set(top10) & flagged)}
        print(f"  {s:<5} NDCG@10 {entry[s]['ndcg10']:.3f} | MAP {entry[s]['map']:.3f} | top 10 graded 2/1/0 = "
              f"{entry[s]['grade2']}/{entry[s]['grade1']}/{entry[s]['grade0']}"
              + (f" | {entry[s]['flagged']} flagged ({flags[qid]['label']})" if flagged else ""))
    judged_summary[qid] = entry

print("\nPre-registered predictions for the judged queries")
pred = {}

# Negation: each system's top 10 contains >= 3 documents mainly about the excluded topic
neg = {(s, q): judged_summary[q][s]["flagged"] for q in ("NF-NEG-01", "NF-NEG-02") for s in SYSTEMS}
neg_ok = all(v >= 3 for v in neg.values())
pred["negation"] = {"counts": {f"{s}/{q}": v for (s, q), v in neg.items()}, "holds": neg_ok}
print(f"  negation: flagged documents in each system's top 10: "
      + ", ".join(f"{s} {q} = {v}" for (s, q), v in neg.items()) + f" -> {'SUPPORTED' if neg_ok else 'NOT SUPPORTED'} (needs >= 3 in all four)")
for q in ("NF-NEG-01", "NF-NEG-02"):
    n_flag = len(flags.get(q, {}).get("docs", []))
    if n_flag < 3:
        print(f"     note: only {n_flag} documents in the {q} pool are mainly about the excluded topic, so >= 3 is impossible there")

# Homonym: dense has fewer wrong-sense documents in its top 10 than BM25, on both queries
hom = {q: (judged_summary[q]["dense"]["flagged"], judged_summary[q]["bm25"]["flagged"]) for q in ("NF-HOM-01", "NF-HOM-02")}
hom_ok = all(d < b for d, b in hom.values())
pred["homonym"] = {"wrong_sense_dense_vs_bm25": hom, "holds": hom_ok}
print("  homonym: wrong-sense documents in top 10 (dense vs BM25): "
      + ", ".join(f"{q} {d} vs {b}" for q, (d, b) in hom.items()) + f" -> {'SUPPORTED' if hom_ok else 'NOT SUPPORTED'}")

# Constraints: fewer than half of each system's top 10 graded 2
con = {(s, q): judged_summary[q][s]["grade2"] for q in ("NF-NUM-01", "NF-TMP-01") for s in SYSTEMS}
con_ok = all(v < 5 for v in con.values())
pred["constraints"] = {"grade2_in_top10": {f"{s}/{q}": v for (s, q), v in con.items()}, "holds": con_ok}
print("  constraints: documents graded 2 in the top 10: "
      + ", ".join(f"{s} {q} = {v}" for (s, q), v in con.items()) + f" -> {'SUPPORTED' if con_ok else 'NOT SUPPORTED'} (needs < 5 in all four)")
for q in ("NF-NUM-01", "NF-TMP-01"):
    n2 = judged_summary[q]["pool_grades"][2]
    if n2 < 5:
        print(f"     note: only {n2} documents in the {q} pool are graded 2, so this prediction cannot fail for {q}")

# ---------------------------------------------------------------------------
# 4. H5 (error-driven): context enrichment of brand-name queries
# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("3. H5 (error-driven): brand name + plain descriptor")
print("=" * 96)
ctx = [r for r in rows if r["category"] == "context_enrichment"]
nf_corpus = load_corpus("data/nfcorpus/corpus.jsonl")
for r in ctx + [x for x in rows if x["category"] == "technical_rewrite"]:
    print(f"\n{r['qid']}  \"{r['source_text']}\" -> \"{r['variant']}\"")
    for s in SYSTEMS:
        top3 = [short(nf_corpus[d]["title"], 55) for d in ranking(runs[s].get(r["qid"], {}))[:3]]
        print(f"  {s:<5} NDCG@10 {r[f'{s}_source']:.3f} -> {r[f'{s}_variant']:.3f} | top 3: {top3 if top3 else 'no results'}")

h5 = {}
if ctx:
    n_dense_pos = sum(1 for r in ctx if r["dense_variant"] > 0)
    h5["H5a"] = {"dense_above_zero": n_dense_pos, "n": len(ctx), "holds": n_dense_pos >= 2}
    bm25_pos = [r["qid"] for r in ctx if r["bm25_variant"] > 0]
    h5["H5b"] = {"bm25_above_zero": bm25_pos, "n": len(ctx), "holds": len(bm25_pos) == len(ctx)}
    gain_d = mean(r["dense_change"] for r in ctx)
    gain_b = mean(r["bm25_change"] for r in ctx)
    h5["H5c"] = {"dense_gain": gain_d, "bm25_gain": gain_b, "holds": gain_d > gain_b}
    print(f"\nH5a (dense above 0 on at least 2 of {len(ctx)}): {n_dense_pos}/{len(ctx)} -> "
          f"{'SUPPORTED' if h5['H5a']['holds'] else 'REFUTED'}")
    print(f"H5b (BM25 above 0 on all {len(ctx)}, since every descriptor word is in the vocabulary): "
          f"{len(bm25_pos)}/{len(ctx)} -> {'SUPPORTED' if h5['H5b']['holds'] else 'REFUTED'}")
    print(f"H5c (dense mean gain > BM25 mean gain): dense {gain_d:+.3f} vs BM25 {gain_b:+.3f} -> "
          f"{'SUPPORTED' if h5['H5c']['holds'] else 'REFUTED'}")

# ---------------------------------------------------------------------------
# 5. Save the summary and a chart for the deck
# ---------------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
with open("runs/suite_summary.json", "w", encoding="utf-8") as f:
    json.dump({"categories": cat_summary, "inherited_rows": rows, "judged": judged_summary,
               "judged_predictions": pred, "H5": h5}, f, indent=1, default=str)
print("\nsaved runs/suite_summary.json and suite/nfcorpus/qrels/full.tsv")

try:
    import matplotlib.pyplot as plt

    cats = list(cat_summary)
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], [cat_summary[c]["bm25"]["change"] for c in cats], 0.4, label="BM25", color="#8A94A0")
    ax.bar([i + 0.2 for i in x], [cat_summary[c]["dense"]["change"] for c in cats], 0.4, label="Dense", color="#1D4E89")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x), [f"{c.replace('_', ' ')}\n(n={cat_summary[c]['n']})" for c in cats], fontsize=8)
    ax.set_ylabel("mean change in NDCG@10\n(variant - original)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/suite_changes.png", dpi=200)
    print("saved figures/suite_changes.png")
except ImportError:
    print("matplotlib not installed: skipped the chart")
