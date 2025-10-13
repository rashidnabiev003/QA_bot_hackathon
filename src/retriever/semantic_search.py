import json
import numpy as np
import faiss
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from src.schemas.pydantic_schemas import BuildConfig

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, workdir: str, config: BuildConfig):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = "BAAI/bge-m3"
        self.rerank_model = "BAAI/bge-m3"
        self.meta_path = self.workdir / "chunks.jsonl"
        self.embed_path = self.workdir / "embeddings.npy"
        self.index_path = self.workdir / "faiss.index"
        self.meta: List[Dict] = []
        self.emb: Optional[np.ndarray] = None
        self.index = None

    def _chunk(self, text: str) -> List[Dict]:
        s = self.cfg.chunk_size
        o = self.cfg.chunk_overlap
        step = max(1, s - o)
        i, out, cid = 0, [], 0
        while i < len(text):
            j = min(len(text), i + s)
            out.append({"id": cid, "start": i, "end": j, "text": text[i:j]})
            cid += 1
            if j == len(text): break
            i += step
        return out

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = self.embedding_model.encode(texts, batch_size=self.config.batch_size)["dence_vecs"]
        v = out["dense_vecs"]
        return self._l2_normalize(v.astype(np.float32))
    
    def build_from_text(self, text: str) -> None:
        chunks = self._chunk(text)
        vecs = self._embed([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(self.index_path))
        self.index = index
        self.meta = chunks
        np.save(self.embed_path, vecs)
        self.meta_path.write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in chunks]), encoding="utf-8")

    def load(self) -> None:
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.emb = None
            return
        except Exception as e:
            logging.warning(f"Failed to load FAISS index: {e}")

    def retrieve(self, query: str, topn: int = 50, topk: int = 5) -> dict:
        q = self._embed([query])
        D, I = self.index.search(q, topn)
        cand = [(i, self.meta[i]["text"]) for i in I[0]]
        pairs = [[query, t] for _, t in cand]
        scores = self.rerank_model.compute_score(pairs)
        s = scores.get("colbert+sparse+dense") or scores.get("sparse+dense") or scores.get("colbert") or scores
        s = np.array(s, dtype="float32") if isinstance(s, list) else np.full(len(cand), float(s), dtype="float32")
        order = np.argsort(-s)[:topk]
        res = [{"id": int(cand[r][0]), "score": float(s[r]), "text": cand[r][1]} for r in order]
        return {"query": query, "results": res}
    