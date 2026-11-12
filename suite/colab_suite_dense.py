# ---- Colab cell: dense retrieval on the 23 suite queries (CPU runtime is fine) ----
# Uses the document embeddings already cached on Drive, so nothing is re-encoded.
from google.colab import drive
drive.mount("/content/drive")
%cd /content
!rm -rf ir-project && git clone -q https://github.com/shail-null-null/ir-project.git
%cd ir-project
!pip install -q transformers pytrec-eval-terrier

import torch
from src.data import load_queries
from src.dense import DenseRetriever
from src.evaluation import write_run

ART = "/content/drive/MyDrive/ir-project-artifacts"
INSTRUCTIONS = {
    "scifact":  "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
}
r = DenseRetriever(batch_size=32)
for ds, instruction in INSTRUCTIONS.items():
    cache = torch.load(f"{ART}/dense_{ds}_docs.pt")
    r.doc_ids, r.doc_matrix = cache["doc_ids"], cache["doc_matrix"]   # cached corpus, same order as when indexed
    r.instruction = instruction
    queries = load_queries(f"suite/{ds}/queries.jsonl")
    run = r.search(queries, k=1000)
    write_run(run, f"{ART}/suite_dense_{ds}.trec", "qwen3-emb-0.6b")
    print(f"{ds}: {len(queries)} suite queries -> {ART}/suite_dense_{ds}.trec")
