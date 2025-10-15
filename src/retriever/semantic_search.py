import json
import re
import numpy as np
import faiss
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
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
        self.embedding_model = BGEM3FlagModel(
                "deepvk/USER-bge-m3",
                device=self.device,
                use_fp16=getattr(self.config, 'use_fp16_rerank', True)
            )
        self.rerank_model = BGEM3FlagModel(
                "BAAI/bge-reranker-v2-m3",
                device=self.device,
                use_fp16=getattr(self.config, 'use_fp16_rerank', True)
            )
        self.meta_path = self.workdir / "chunks.jsonl"
        self.embed_path = self.workdir / "embeddings.npy"
        self.index_path = self.workdir / "faiss.index"
        self.kb_meta = self.workdir / "kb.json"
        self.meta: List[Dict] = []
        self.emb: Optional[np.ndarray] = None
        self.index = None

    def _sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        s = re.split(r"(?<=[\.\!\?…])\s+(?=[A-ZА-ЯЁ0-9«(])", text)
        return [t.strip() for t in s if t.strip()]
    
    def _chunk(self, text: str) -> List[Dict[str, Any]]:
            sents = self._sentences(text)
            out, buf, clen = [], [], 0
            for sent in sents:
                if buf and clen + len(sent) > self.config.chunk_size:
                    out.append({"id": len(out), "text": " ".join(buf)})
                    over = []
                    while buf and sum(len(x) for x in over) < self.config.overlap_size:
                        over.insert(0, buf.pop())
                    buf = over + [sent]
                    clen = sum(len(x) for x in buf)
                else:
                    buf.append(sent); clen += len(sent)
            if buf:
                out.append({"id": len(out), "text": " ".join(buf)})
            return out

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = self.embedding_model.encode(texts, batch_size=self.config.batch_size)
        vecs = out["dense_vecs"]
        return self._l2_normalize(vecs)

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
        self.kb_meta.write_text(json.dumps({"dim": int(vecs.shape[1]), "count": len(chunks)}, ensure_ascii=False), encoding="utf-8")


    def load(self) -> None:
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.meta = [json.loads(x) for x in self.meta_path.read_text(encoding="utf-8").splitlines()]
            self.emb = None
            return
        except Exception as e:
            logging.warning(f"Failed to load FAISS index: {e}")

    def retrieve(self, query: str, topn: int = 50, topk: int = 5) -> Dict:
        if self.index is None or not self.meta:
            self.load()
        qv = self._embed([query])
        D, I = self.index.search(qv, topn) 
        cand = [(int(i), self.meta[int(i)]["text"], float(D[0, k])) for k, i in enumerate(I[0])]
        if self.config.enable_rerank:
            pairs = [[query, t] for _, t, _ in cand]
            out = self.rerank_model.compute_score(pairs)  
            s = out.get("colbert+sparse+dense") or out.get("sparse+dense") or out.get("colbert") or out
            if isinstance(s, list):
                scores = np.asarray(s, dtype="float32")
            else:
                scores = np.full(len(cand), float(s), dtype="float32")
            order = np.argsort(-scores)[:topk]
            res = [{"id": int(cand[i][0]), "score_rerank": float(scores[i]), "score_dense": float(cand[i][2]), "text": cand[i][1]} for i in order]
        else:
            order = np.argsort(-D[0])[:topk]
            res = [{"id": int(I[0][i]), "score_dense": float(D[0][i]), "text": self.meta[int(I[0][i])]["text"]} for i in order]
        return {"query": query, "results": res}
    