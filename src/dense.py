"""Dense retrieval with Qwen3-Embedding: embed documents and queries, exact search."""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseRetriever:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", instruction: str = "",
                 batch_size: int = 16, max_length: int = 512):
        # GPU with float16 if available; otherwise CPU with float32 (float16 is slow or unsupported on CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()

        self.instruction = instruction   # prepended to queries only, never to documents
        self.batch_size = batch_size
        self.max_length = max_length

        self.doc_ids = []                # row i of doc_matrix belongs to doc_ids[i]
        self.doc_matrix = None           # (N, hidden), float32, on CPU, filled by index()

    def embed(self, texts: list[str]) -> torch.Tensor:
        """L2-normalised last-token embeddings, shape (len(texts), hidden), float32 on CPU."""
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state          # (b, seq_len, hidden)
            last = batch["attention_mask"].sum(dim=1) - 1                # last real token per row (right padding)
            rows = torch.arange(hidden.shape[0], device=self.device)
            vectors = hidden[rows, last].float()                         # (b, hidden), float32 before normalising
            chunks.append(F.normalize(vectors, p=2, dim=1).cpu())        # move off the GPU as we go
        return torch.cat(chunks, dim=0)

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        """Embed every document as title + " " + text. Row i of doc_matrix is doc_ids[i]."""
        self.doc_ids = list(corpus)
        texts = [f"{corpus[d]['title']} {corpus[d]['text']}" for d in self.doc_ids]
        self.doc_matrix = self.embed(texts)

    def search(self, queries: dict[str, str], k: int = 1000) -> dict[str, dict[str, float]]:
        """Return a run {qid: {doc_id: score}} with the top-k documents per query."""
        # 1. fixed-order lists: texts[i] is the formatted version of qids[i]
        qids = list(queries.keys())
        texts = []
        for qid in qids:
            q = f"Instruct: {self.instruction}\nQuery:{queries[qid]}"
            texts.append(q)

        # 2. embed queries, score against every document: (n_queries, n_docs)
        Q = self.embed(texts)
        scores = Q @ self.doc_matrix.T

        # 3. top-k per query: values and column positions, each (n_queries, k)
        val, pos = torch.topk(scores, min(k, scores.shape[1]), dim=1)

        # 4. positions -> document IDs; row i belongs to qids[i]
        run = {}
        for i, score_val in enumerate(val):
            dic = {}
            for score, position in zip(score_val.tolist(), pos[i].tolist()):
                dic[self.doc_ids[position]] = score
            run[qids[i]] = dic
        return run
