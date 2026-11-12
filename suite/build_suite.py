"""Build the stress-test suite files from sources.json and judged_queries.json.

Run from the project root:   python -m suite.build_suite
Writes, per dataset (scifact, nfcorpus):
  suite/<dataset>/queries.jsonl       all suite queries for that dataset
  suite/<dataset>/qrels/test.tsv      inherited labels (judged queries are added after judging)
and suite/metadata.json             id -> category, source query, error_driven, judged
"""
import json
import os

from src.data import load_qrels

PREFIX = {"identifier_only": "IDN", "negation_flip": "NEG", "lay_rewrite": "LAY",
          "technical_rewrite": "TEC", "context_enrichment": "CTX"}
DS_PREFIX = {"scifact": "SF", "nfcorpus": "NF"}

with open("suite/sources.json", encoding="utf-8") as f:
    sources = json.load(f)["sources"]
with open("suite/judged_queries.json", encoding="utf-8") as f:
    judged = json.load(f)["queries"]

queries = {"scifact": {}, "nfcorpus": {}}
qrels = {"scifact": {}, "nfcorpus": {}}
metadata = {}
original_qrels = {ds: load_qrels(f"data/{ds}/qrels/test.tsv") for ds in queries}
counters = {}

for s in sources:
    ds, cat = s["dataset"], s["category"]
    assert s["variant"].strip(), f"empty variant for {s['source_qid']}"
    key = (ds, cat)
    counters[key] = counters.get(key, 0) + 1
    qid = f"{DS_PREFIX[ds]}-{PREFIX[cat]}-{counters[key]:02d}"
    queries[ds][qid] = s["variant"]
    qrels[ds][qid] = dict(original_qrels[ds][s["source_qid"]])   # inherited labels
    metadata[qid] = {"dataset": ds, "category": cat, "source_qid": s["source_qid"],
                     "source_text": s["source_text"], "text": s["variant"],
                     "error_driven": bool(s.get("error_driven")), "judged": False}

for j in judged:
    queries["nfcorpus"][j["id"]] = j["text"]
    metadata[j["id"]] = {"dataset": "nfcorpus", "category": j["category"], "source_qid": None,
                         "text": j["text"], "error_driven": False, "judged": True}

for ds in queries:
    os.makedirs(f"suite/{ds}/qrels", exist_ok=True)
    with open(f"suite/{ds}/queries.jsonl", "w", encoding="utf-8") as f:
        for qid, text in queries[ds].items():
            f.write(json.dumps({"_id": qid, "text": text, "metadata": {}}, ensure_ascii=False) + "\n")
    with open(f"suite/{ds}/qrels/test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, rels in qrels[ds].items():
            for doc_id, grade in rels.items():
                f.write(f"{qid}\t{doc_id}\t{grade}\n")
    print(f"{ds}: {len(queries[ds])} queries ({len(qrels[ds])} with inherited labels)")

with open("suite/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"total {len(metadata)} suite queries; saved suite/metadata.json")
