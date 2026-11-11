"""Charts for the deck, from runs/analysis_summary.json (written by compare.py).

Run from the project root:   python make_charts.py
Saves figures/results_ndcg10.png and figures/perquery_diffs.png.
"""
import json
import os

import matplotlib.pyplot as plt

DATASETS = ["scifact", "nfcorpus", "scidocs"]
LABELS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "scidocs": "SciDocs"}
BM25_COLOR, DENSE_COLOR = "#8A94A0", "#1D4E89"

with open("runs/analysis_summary.json", encoding="utf-8") as f:
    s = json.load(f)
os.makedirs("figures", exist_ok=True)

# 1. NDCG@10, BM25 vs dense, per dataset
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(DATASETS))
w = 0.38
bm25 = [s[d]["bm25"] for d in DATASETS]
dense = [s[d]["dense"] for d in DATASETS]
b1 = ax.bar([i - w / 2 for i in x], bm25, w, label="BM25", color=BM25_COLOR)
b2 = ax.bar([i + w / 2 for i in x], dense, w, label="Dense (Qwen3-Embedding-0.6B)", color=DENSE_COLOR)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(list(x), [LABELS[d] for d in DATASETS])
ax.set_ylabel("NDCG@10")
ax.set_ylim(0, max(dense + bm25) * 1.18)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/results_ndcg10.png", dpi=200)

# 2. Distribution of per-query differences (dense - BM25)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
for ax, d in zip(axes, DATASETS):
    ax.hist(s[d]["diffs"], bins=30, color=DENSE_COLOR, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{LABELS[d]}: dense better on {s[d]['dense_wins']}, "
                 f"BM25 on {s[d]['bm25_wins']}", fontsize=9)
    ax.set_xlabel("NDCG@10: dense − BM25")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("queries")
fig.tight_layout()
fig.savefig("figures/perquery_diffs.png", dpi=200)
print("saved figures/results_ndcg10.png and figures/perquery_diffs.png")
