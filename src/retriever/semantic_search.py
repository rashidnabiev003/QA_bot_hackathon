import os
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
import numpy as np
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _load_bm25_from_jsonl(jsonl_path: Path, k: int = 50) -> BM25Retriever:
    rows: List[Document] = []
    if not jsonl_path.exists():
        return BM25Retriever.from_documents(rows)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        rows.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata") or {}))
    bm25 = BM25Retriever.from_documents(rows)
    bm25.k = k
    return bm25


class SemanticSearch:
    """Класс для  семантического поиска с гибридным (FAISS+dense + BM25) и CrossEncoder rerank."""

    def __init__(self, index_dir: Optional[str] = None, build_config: Optional[Any] = None) -> None:
        self.index_dir: Path = Path(index_dir or os.getenv("INDEX_DIR", "src/data")).resolve()
        self.faiss_dir: Path = self.index_dir / "faiss_user_bge_m3"
        self.bm25_jsonl: Path = self.index_dir / "bm25_corpus.jsonl"

        # Конфиг
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL")
        self.rerank_model: str = os.getenv("RERANK_MODEL")
        self.emb_on_cuda: bool = torch.cuda.is_available()

        self.chunk_size: int = int(getattr(build_config, "chunk_size", os.getenv("CHUNK_SIZE", 128)))
        self.chunk_overlap: int = int(getattr(build_config, "overlap_size", os.getenv("CHUNK_OVERLAP", 50)))

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._tokenizer = None
        self._faiss: Optional[FAISS] = None

        self.ready: bool = False
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.reranker = FlagReranker(
            self.rerank_model,
            device="cpu",
        )

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            device = "cuda" if self.emb_on_cuda else "cpu"
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _split_text(self, text: str) -> List[Document]:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, use_fast=True)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        base_doc = Document(page_content=text, metadata={"source": "docx"})
        return splitter.split_documents([base_doc])

    def build_from_text(self, text: str) -> None:
        """Строит индекс с нуля: FAISS + BM25 jsonl."""
        try:
            docs = self._split_text(text)
            emb = self._get_embeddings()
            vs = FAISS.from_documents(docs, emb)
            vs.save_local(str(self.faiss_dir))
            self._faiss = vs

            # Сохраняем BM25 корпус в jsonl
            self.bm25_jsonl.write_text(
                "\n".join(
                    json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False)
                    for d in docs
                ),
                encoding="utf-8",
            )
            logger.info(
                {
                    "event": "index_built",
                    "faiss_dir": str(self.faiss_dir),
                    "bm25_corpus": str(self.bm25_jsonl),
                    "n_chunks": len(docs),
                    "embed": self.embedding_model,
                }
            )
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            self.ready = False

    def load(self) -> None:
        """Загружает индекс из self.index_dir."""
        try:
            emb = self._get_embeddings()
            if not self.faiss_dir.exists():
                logger.warning(f"FAISS dir not found: {self.faiss_dir}")
                self.ready = False
                return
            self._faiss = FAISS.load_local(str(self.faiss_dir), emb, allow_dangerous_deserialization=True)
            self.ready = True
            logger.info({"event": "index_loaded", "faiss_dir": str(self.faiss_dir)})
        except Exception as e:
            logger.error(f"Failed to load FAISS: {e}")
            self.ready = False

    def _make_retriever(self, top_k_dense: int, top_k_final: int) -> ContextualCompressionRetriever:
        if not self._faiss:
            raise RuntimeError("FAISS is not loaded. Call load() or build_from_text() first.")
        dense = self._faiss.as_retriever(search_kwargs={"k": top_k_dense})
        bm25 = _load_bm25_from_jsonl(self.bm25_jsonl, k=top_k_dense)
        hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.7, 0.3])
        device = "cuda" if (self.emb_on_cuda and torch.cuda.is_available()) else "cpu"
        ce = HuggingFaceCrossEncoder(model_name=self.rerank_model, model_kwargs={"device": device})
        compressor = CrossEncoderReranker(model=ce, top_n=top_k_final)

        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid)

    def retrieve(self, query: str, topn: int = 40, topk: int = 8) -> Dict[str, Any]:
        if not self.ready:
            self.load()
        if not self.ready:
            return {"results": [], "meta": {"error": "index_not_ready"}}

        retr = self._make_retriever(top_k_dense=topn, top_k_final=topk)
        docs = retr.get_relevant_documents(query)

        score_rerank: List[float]
        try:
            pairs = [[query, d.page_content] for d in docs]
            score_rerank = self.reranker.compute_score(pairs, normalize=True)
            if not isinstance(score_rerank, list):
                score_rerank = [float(score_rerank)] * len(docs)
        except Exception as e:
            logger.warning("FlagReranker failed: %s", e)
            score_rerank = [0.0] * len(docs)

        logger.info({'event': 'ce_scores_head', 'scores': score_rerank[:5]})

        if score_rerank and len(score_rerank) == len(docs):
            order = np.argsort(-np.asarray(score_rerank))
            docs = [docs[i] for i in order]
            score_rerank = [score_rerank[i] for i in order]

        results: List[Dict[str, Any]] = []
        for i, d in enumerate(docs):
            md = dict(d.metadata or {})
            s = float(score_rerank[i]) if i < len(score_rerank) else 0.0
            md["relevance_score"] = s  
            results.append({
                "text": d.page_content,
                "metadata": md,
                "score": s, 
            })

        return {
            "results": results,
            "meta": {"topn": topn, "topk": topk, "n_returned": len(results)},
        }

    def clear_index(self) -> None:
        """Удаляет сохранённые артефакты индекса."""
        try:
            if self.faiss_dir.exists():
                shutil.rmtree(self.faiss_dir, ignore_errors=True)
            if self.bm25_jsonl.exists():
                self.bm25_jsonl.unlink(missing_ok=True)  
            self.ready = False
            logger.info({"event": "index_cleared", "dir": str(self.index_dir)})
        except Exception as e:
            logger.warning(f"Failed to clear index: {e}")
