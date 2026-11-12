"""Step 1 of the stress-test suite: select the source queries for the 14 inherited-label variants.

Selection is by rule with a fixed seed, never by hand, so the choice can't be biased by
results we have already seen. Run from the project root:   python suite/select_sources.py
Writes suite/sources.json; the "variant" fields are filled in by hand in step 2.
"""
import json
import random
import re
import sys

sys.path.insert(0, ".")   # so `src` imports work when run as `python suite/select_sources.py`
from src.data import filter_queries, load_qrels, load_queries  # noqa: E402

SEED = 2026

# The five brand-name queries from the H1 vocabulary-gap set (IDs from inspect_empty.py)
BRAND_NAME_QIDS = {"PLAIN-520": "alli", "PLAIN-1214": "fosamax", "PLAIN-1621": "mevacor",
                   "PLAIN-2134": "splenda", "PLAIN-2408": "zoloft"}


def has_identifier(text):
    """Same rule as H3 in compare.py: a token containing both a letter and a digit."""
    tokens = re.findall(r"[A-Za-z0-9-]+", text)
    return any(re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t) for t in tokens)


def has_not(text):
    """The word "not" on its own (so "notable" or "knot" don't count)."""
    return re.search(r"\bnot\b", text, flags=re.IGNORECASE) is not None


def has_any_negation(text):
    return re.search(r"\b(not|no|never|without|lack|lacks|lacking)\b", text, flags=re.IGNORECASE) is not None


def test_queries(dataset):
    qrels = load_qrels(f"data/{dataset}/qrels/test.tsv")
    return filter_queries(load_queries(f"data/{dataset}/queries.jsonl"), qrels)


rng = random.Random(SEED)
scifact = test_queries("scifact")
nfcorpus = test_queries("nfcorpus")
sf_ids = sorted(scifact)          # sorted first, so the sample doesn't depend on file order
used = set()
selected = []


def take(category, dataset, pool, n, change, queries):
    """Sample n unused query IDs from pool and record them."""
    pool = [q for q in pool if q not in used]
    if len(pool) < n:
        raise SystemExit(f"{category}: only {len(pool)} candidates for {n} slots")
    for qid in rng.sample(pool, n):
        used.add(qid)
        selected.append({"category": category, "dataset": dataset, "source_qid": qid,
                         "source_text": queries[qid], "change": change, "variant": ""})


# 1. Identifier-only (4): claims with an identifier, cut down to the identifier + 1-2 key words
take("identifier_only", "scifact", [q for q in sf_ids if has_identifier(scifact[q])], 4,
     "reduce to the identifier plus 1-2 key words", scifact)

# 2. Negation flip (5): remove "not" from negated claims; top up by adding "not" if needed
negated = [q for q in sf_ids if has_not(scifact[q]) and q not in used]
n_remove = min(5, len(negated))
take("negation_flip", "scifact", negated, n_remove, "remove the negation", scifact)
if n_remove < 5:
    positive = [q for q in sf_ids if not has_any_negation(scifact[q]) and not has_identifier(scifact[q])]
    take("negation_flip", "scifact", positive, 5 - n_remove, "add a negation", scifact)

# 3. Lay rewrite (3): claims without identifiers or negation, rewritten in everyday language
plain = [q for q in sf_ids if not has_identifier(scifact[q]) and not has_any_negation(scifact[q])]
take("lay_rewrite", "scifact", plain, 3, "rewrite in lay language, same meaning", scifact)

# 4. Technical rewrite (2): brand-name NFCorpus queries -> the generic/technical name
for qid, name in BRAND_NAME_QIDS.items():   # guard against wrong IDs
    if qid not in nfcorpus or nfcorpus[qid].strip().lower() != name:
        raise SystemExit(f"{qid}: expected '{name}', got {nfcorpus.get(qid)!r}")
take("technical_rewrite", "nfcorpus", sorted(BRAND_NAME_QIDS), 2,
     "replace the brand name with the generic / technical name", nfcorpus)

print(f"seed = {SEED}; {sum(has_not(scifact[q]) for q in sf_ids)} SciFact test claims contain 'not'\n")
for s in selected:
    print(f"{s['category']:<18} {s['dataset']:<9} {s['source_qid']:<11} {s['source_text'][:80]}")

with open("suite/sources.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "sources": selected}, f, indent=2, ensure_ascii=False)
print(f"\nsaved {len(selected)} sources to suite/sources.json")
