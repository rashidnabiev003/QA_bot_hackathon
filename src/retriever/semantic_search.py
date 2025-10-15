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
        self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", device=self.device, use_fp16=config.use_fp16_rerank if hasattr(config, 'use_fp16_rerank') else True)
        self.rerank_model = BGEM3FlagModel("BAAI/bge-m3", device=config.rerank_device, use_fp16=config.use_fp16_rerank)
        base_dir = Path("src/data")
        base_dir.mkdir(parents=True, exist_ok=True)
        # хранить данные индекса внутри src/data
        self.meta_path = base_dir / "chunks.jsonl"
        self.embed_path = base_dir / "embeddings.npy"
        self.index_path = base_dir / "faiss.index"
        self.meta: List[Dict] = []
        self.emb: Optional[np.ndarray] = None
        self.index = None

    def _chunk(self, text: str) -> List[Dict]:
        s = self.config.chunk_size
        o = self.config.overlap_size
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
        if isinstance(self.embedding_model, BGEM3FlagModel):
            out = self.embedding_model.encode(texts, batch_size=self.config.batch_size)
            v = out["dense_vecs"]
            return self._l2_normalize(v.astype(np.float32))
        else:
            v = self.embedding_model.encode(texts, batch_size=self.config.batch_size)
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
        cand = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.meta):
                continue
            cand.append((int(idx), self.meta[idx]["text"]))
        if not cand:
            return {"query": query, "results": []}
        pairs = [[query, t] for _, t in cand]
        scores = self.rerank_model.compute_score(pairs)
        s = scores.get("colbert+sparse+dense") or scores.get("sparse+dense") or scores.get("colbert") or scores
        s = np.array(s, dtype="float32") if isinstance(s, list) else np.full(len(cand), float(s), dtype="float32")
        order = np.argsort(-s)
        topk = min(topk, len(order))
        res = [{"id": int(cand[r][0]), "score": float(s[r]), "text": cand[r][1]} for r in order[:topk]]
        return {"query": query, "results": res}
    