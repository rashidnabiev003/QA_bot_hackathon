import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.schemas.pydantic_schemas import BuildConfig, MetricConfig
from src.retriever.semantic_search import SemanticSearch
from src.retriever.metrics import MetricComputer
from src.llm.llm_reranker_vllm import rerank_with_llm
from src.llm.llm_chat_bot_ollama import generate_answer

logger = logging.getLogger(__name__)

class QAPipeline:
    def __init__(self, index_dir: str = "./index"):
        self.index_dir = index_dir
        self.build_config = BuildConfig(
            batch_size=32,
            force_cpu=False,
            rerank_device=os.getenv("RERANK_DEVICE", "cuda"),
            use_fp16_rerank=True,
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            overlap_size=int(os.getenv("CHUNK_OVERLAP", "50")),
        )
        self.metric_config = MetricConfig(
            bleurt_ckpt=os.getenv("BLEURT_CHECKPOINT", "BLEURT-20"),
            sas_model=os.getenv("SAS_MODEL", "BAAI/bge-m3"),
            sas_device=os.getenv("SAS_DEVICE", "cuda"),
            sas_fp16=True,
            bleurt_endpoint=os.getenv("BLEURT_URL"),
        )
        self.engine = SemanticSearch(self.index_dir, self.build_config)
        self._ensure_index_loaded()
        self.mc = MetricComputer(self.metric_config)

    def _ensure_index_loaded(self) -> None:
        try:
            self.engine.load()
            if self.engine.index is None:
                self._maybe_build_index_from_docx()
                self.engine.load()
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._maybe_build_index_from_docx()
            self.engine.load()

    def _maybe_build_index_from_docx(self) -> None:
        docx_path = os.getenv("DOCX_PATH", str(Path("src/data/input.docx").resolve()))
        if not Path(docx_path).exists():
            logger.warning(f"DOCX file not found: {docx_path}")
            return
        try:
            from src.utils.file_loader import parse_docx_to_text
            text = parse_docx_to_text(docx_path)
            if not text:
                logger.warning("Parsed DOCX is empty")
                return
            self.engine.build_from_text(text)
            logger.info("Index built from DOCX")
        except Exception as e:
            logger.error(f"Failed to build index from DOCX: {e}")

    async def answer(self, query: str, topn: int = 50, topk: int = 5, use_llm_rerank: Optional[bool] = None) -> Dict[str, Any]:
        if use_llm_rerank is None:
            use_llm_rerank = os.getenv("USE_LLM_RERANK", "0") in ("1", "true", "True")
        result = self.engine.retrieve(query, topn=topn, topk=max(topk, 10))
        contexts = result.get("results", [])
        if use_llm_rerank and contexts:
            contexts = await rerank_with_llm(
                query=query,
                chunks=contexts,
                topk=topk,
                vllm_url=os.getenv("VLLM_URL", "http://localhost:8000"),
                model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Thinking-2507"),
            )
        contexts = contexts[:topk]
        answer = await generate_answer(
            query=query,
            contexts=contexts,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "open-ai/gpt-oss-20b"),
        )
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
        }

pipeline_singleton: Optional[QAPipeline] = None

def get_pipeline() -> QAPipeline:
    global pipeline_singleton
    if pipeline_singleton is None:
        pipeline_singleton = QAPipeline()
    return pipeline_singleton

