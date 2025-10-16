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
from transformers import AutoTokenizer 

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, workdir: str, config: BuildConfig):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.meta_path = self.workdir / "chunks.jsonl"
        self.embed_path = self.workdir / "embeddings.npy"
        self.index_path = self.workdir / "faiss.index"
        self.kb_meta = self.workdir / "kb.json"
        self.meta: List[Dict] = []
        self.emb: Optional[np.ndarray] = None
        self.index = None

        logger.info("Loading embedding model: BAAI/bge-m3 on %s (fp16=%s)", self.device, getattr(self.config, 'use_fp16_rerank', True))
        self.embedding_model = SentenceTransformer(
                "deepvk/USER-bge-m3",
                device=self.device,
            )

        self.rerank_model = BGEM3FlagModel(
                    "BAAI/bge-reranker-v2-m3",
                    device=self.device,
                    use_fp16=getattr(self.config, 'use_fp16_rerank', True)
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(
                self.config,
                "tokenizer_name",
                  getattr(self.config, "embed_model", "deepvk/USER-bge-m3")
                  ),
            use_fast=True
            )
        
    def _tok_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_tokens(self, text: str, max_toks: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_toks:
            return text
        return self.tokenizer.decode(ids[:max_toks], skip_special_tokens=True)
        
    def _sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        s = re.split(r"(?<=[\.\!\?…])\s+(?=[A-ZА-ЯЁ0-9«(])", text)
        return [t.strip() for t in s if t.strip()]
    
    def _chunk(self, text: str):
        max_t = self.config.chunk_size
        over_t = self.config.overlap_size
        sents = [t.strip() for t in re.split(r"(?<=[\.\!\?…])\s+(?=[A-ZА-ЯЁ0-9«(])", re.sub(r"\s+", " ", text).strip()) if t.strip()]
        out, buf, tokc = [], [], 0
        for s in sents:
            ls = self._tok_len(s)
            if buf and tokc + ls > max_t:
                ch = " ".join(buf)
                out.append({"id": len(out), "text": ch, "tok_len": tokc})
                keep, keep_len = [], 0
                for ss in reversed(buf):
                    lss = self._tok_len(ss)
                    if keep_len + lss > over_t: break
                    keep.insert(0, ss); keep_len += lss
                buf, tokc = keep + [s], keep_len + ls
            else:
                buf.append(s); tokc += ls
        if buf:
            out.append({"id": len(out), "text": " ".join(buf), "tok_len": tokc})
        return out

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def _embed(self, texts: list[str]) -> np.ndarray:
        max_embed_t = int(getattr(self.config, "embed_max_tokens", getattr(self.config, "chunk_size", 256)))
        prep = [self._truncate_tokens(t, max_embed_t) for t in texts]
        out = self.embedding_model.encode(prep, batch_size=self.config.batch_size)
        vecs = out.get("dense_vecs", out) if isinstance(out, dict) else out
        vecs = np.asarray(vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def build_from_text(self, text: str) -> None:
        logger.info("Chunking text for index build: chunk_size=%s, overlap=%s", self.config.chunk_size, self.config.overlap_size)
        chunks = self._chunk(text)
        vecs = self._embed([c["text"] for c in chunks])
        logger.info("Building FAISS index with dim=%s, vectors=%s", int(vecs.shape[1]), len(chunks))
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(self.index_path))
        self.index = index
        self.meta = chunks
        np.save(self.embed_path, vecs)
        self.meta_path.write_text("\n".join([json.dumps(x, ensure_ascii=False) for x in chunks]), encoding="utf-8")
        self.kb_meta.write_text(json.dumps({"dim": int(vecs.shape[1]), "count": len(chunks)}, ensure_ascii=False), encoding="utf-8")
        logger.info("Index written: %s; Chunks: %s; Embeddings: %s", self.index_path, self.meta_path, self.embed_path)


    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"No FAISS index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"No metadata: {self.meta_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.meta = [json.loads(x) for x in self.meta_path.read_text(encoding="utf-8").splitlines()]
        if len(self.meta) != self.index.ntotal:
            raise ValueError(f"meta/index mismatch: {len(self.meta)} vs {self.index.ntotal}")

    def retrieve(self, query: str, topn: int = 50, topk: int = 5) -> Dict:
        if self.index is None or not self.meta:
            self.load()
        qv = self._embed([query])
        D, I = self.index.search(qv, topn) 
        cand = [(int(i), self.meta[int(i)]["text"], float(D[0, k])) for k, i in enumerate(I[0])]
        if getattr(self.config, "enable_rerank", True):
            q_max = 128
            d_max = 384
            q_r = self._truncate_tokens(query, q_max)
            pairs = [[q_r, self._truncate_tokens(t, d_max)] for _, t, _ in cand]
            scores = self.rerank_model.compute_score(pairs)
        if getattr(self.config, 'enable_rerank', True):
            pairs = [[query, t] for _, t, _ in cand]
            out = self.rerank_model.compute_score(pairs)  
            s = out.get("colbert+sparse+dense") or out.get("sparse+dense") or out.get("colbert") or out
            if isinstance(s, list):
                scores = np.asarray(s, dtype="float32")
            else:
                scores = np.full(len(cand), float(s), dtype="float32")
            order = np.argsort(-scores)[:topk]
            res = [{"id": int(cand[i][0]), "score": float(scores[i]), "text": cand[i][1]} for i in order]
        else:
            order = np.argsort(-D[0])[:topk]
            res = [{"id": int(I[0][i]), "score": float(D[0][i]), "text": self.meta[int(I[0][i])]["text"]} for i in order]
        return {"query": query, "results": res}
    