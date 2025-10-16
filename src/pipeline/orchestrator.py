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
    def __init__(self, index_dir: str = None):
        if index_dir is None:
            index_dir = os.getenv("INDEX_DIR", "src/data")
        self.index_dir = index_dir
        self.build_config = BuildConfig(
            batch_size=16,
            force_cpu=False,
            rerank_device=os.getenv("RERANK_DEVICE", "cuda"),
            use_fp16_rerank=True,
            chunk_size=128,
            overlap_size=50,
        )
        self.metric_config = MetricConfig(
            bleurt_ckpt=os.getenv("BLEURT_CHECKPOINT", "BLEURT-20"),
            sas_model=os.getenv("SAS_MODEL", "BAAI/bge-reranker-v2-m3"),
            sas_device=os.getenv("SAS_DEVICE", "cuda"),
            sas_fp16=True,
            bleurt_endpoint=os.getenv("BLEURT_URL"),
        )
        self.engine = SemanticSearch(self.index_dir, self.build_config)
        self._index_checked = False
        self.mc = MetricComputer(self.metric_config)
        logger.info(
            "QAPipeline init: index_dir=%s, batch_size=%s, chunk_size=%s, overlap_size=%s",
            self.index_dir, self.build_config.batch_size, self.build_config.chunk_size, self.build_config.overlap_size
        )
        logger.info(
            "Metrics config: sas_model=%s, bleurt_endpoint=%s",
            self.metric_config.sas_model, self.metric_config.bleurt_endpoint
        )

    def _ensure_index_loaded(self) -> None:
        try:
            # Опция принудительной пересборки индекса при старте
            force_rebuild = os.getenv("FORCE_REBUILD_INDEX", "0") in ("1", "true", "True")
            if force_rebuild:
                logger.info("FORCE_REBUILD_INDEX=1 — rebuilding index on startup")
                self.rebuild_index()
                return

            self.engine.load()
            if self.engine.index is None:
                auto = os.getenv("AUTO_BUILD_INDEX", "0") in ("1", "true", "True")
                if auto:
                    self._maybe_build_index_from_docx()
                self.engine.load()
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            auto = os.getenv("AUTO_BUILD_INDEX", "0") in ("1", "true", "True")
            if auto:
                self._maybe_build_index_from_docx()
            self.engine.load()

    def rebuild_index(self) -> None:
        """Удаляет старый индекс и пересобирает его из DOCX."""
        try:
            # Удаляем файлы индекса, если существуют
            for p in [self.engine.index_path, self.engine.embed_path, self.engine.meta_path, getattr(self.engine, 'kb_meta', None)]:
                if p is None:
                    continue
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove {p}: {e}")
            # Пересобираем
            self._maybe_build_index_from_docx()
            self.engine.load()
            logger.info("Index rebuilt successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")

    def _maybe_build_index_from_docx(self) -> None:
        docx_path = os.getenv("DOCX_PATH", str(Path("src/data/input.docx").resolve()))
        if not Path(docx_path).exists():
            logger.warning(f"DOCX file not found: {docx_path}")
            return
        try:
            logger.info("Building index from DOCX: %s", docx_path)
            from src.utils.file_loader import parse_docx_to_text
            text = parse_docx_to_text(docx_path)
            if not text:
                logger.warning("Parsed DOCX is empty")
                return
            self.engine.build_from_text(text)
            logger.info("Index built from DOCX → index_path=%s, chunks_path=%s", self.engine.index_path, self.engine.meta_path)
        except Exception as e:
            logger.error(f"Failed to build index from DOCX: {e}")

    async def answer(self, query: str, topn: int = 50, topk: int = 5, use_llm_rerank: Optional[bool] = None) -> Dict[str, Any]:
        import time
        t_start = time.monotonic()
        if not self._index_checked:
            self._ensure_index_loaded()
            self._index_checked = True
        if use_llm_rerank is None:
            use_llm_rerank = os.getenv("USE_LLM_RERANK", "0") in ("1", "true", "True")
        logger.info("Answer pipeline: topn=%s, topk=%s, llm_rerank=%s", topn, topk, use_llm_rerank)
        # Для LLM реранкинга берем в 2 раза больше контекстов чем нужно
        rerank_pool_size = topk * 2 if use_llm_rerank else max(topk, 10)
        t0 = time.monotonic()
        result = self.engine.retrieve(query, topn=topn, topk=rerank_pool_size)
        logger.info("Retrieve done in %.3fs", time.monotonic() - t0)
        contexts = result.get("results", [])
        logger.info(f"Retrieved {len(contexts)} contexts from semantic search")
        if use_llm_rerank and contexts:
            logger.info(f"Applying LLM reranking on {len(contexts)} contexts to select top {topk}")
            t1 = time.monotonic()
            contexts = await rerank_with_llm(
                query=query,
                chunks=contexts,
                topk=topk,
                vllm_url=os.getenv("VLLM_URL", "http://localhost:8000"),
                model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Thinking-2507"),
            )
            logger.info("LLM reranking done in %.3fs", time.monotonic() - t1)
            logger.info(f"After LLM reranking: {len(contexts)} contexts, first scores: {[c.get('llm_score', 0) for c in contexts[:3]]}")
        else:
            contexts = contexts[:topk]
        logger.info(f"Final contexts count: {len(contexts)}")
        t2 = time.monotonic()
        answer = await generate_answer(
            query=query,
            contexts=contexts,
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        )
        logger.info("LLM answer generated in %.3fs", time.monotonic() - t2)
        
        # Извлекаем текст ответа
        answer_text = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
        
        # Вычисляем метрики
        try:
            t3 = time.monotonic()
            metrics_result = await asyncio.to_thread(
                self.mc.compute_all,
                query=query,
                answer=answer_text,
                contexts=[c["text"] for c in contexts],
                ground_truth=None
            )
            logger.info("Metrics computed in %.3fs", time.monotonic() - t3)
            
            # Добавляем метрики в каждый контекст
            for i, ctx in enumerate(contexts):
                ctx["bleurt"] = metrics_result.get("bleurt_per_context", [0])[i] if i < len(metrics_result.get("bleurt_per_context", [])) else 0
                ctx["sas"] = metrics_result.get("sas_per_context", [0])[i] if i < len(metrics_result.get("sas_per_context", [])) else 0
            
            metrics = {
                "confidence": answer.get("confidence", "unknown") if isinstance(answer, dict) else "unknown",
                "sources_used": len(answer.get("sources_used", []) if isinstance(answer, dict) else []),
                "contexts_found": len(contexts),
                "bleurt_avg": metrics_result.get("bleurt_avg", 0),
                "sas_avg": metrics_result.get("sas_avg", 0),
                "rouge_l": metrics_result.get("rouge_l", 0),
                "ndcg": metrics_result.get("ndcg", 0),
                "mrr": metrics_result.get("mrr", 0),
                "map": metrics_result.get("map", 0),
            }
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            metrics = {
                "confidence": answer.get("confidence", "unknown") if isinstance(answer, dict) else "unknown",
                "sources_used": len(answer.get("sources_used", []) if isinstance(answer, dict) else []),
                "contexts_found": len(contexts),
                "error": "Не удалось вычислить метрики"
            }
        
        out = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "metrics": metrics,
        }
        logger.info("Answer pipeline total time: %.3fs", time.monotonic() - t_start)
        return out

pipeline_singleton: Optional[QAPipeline] = None

def get_pipeline() -> QAPipeline:
    global pipeline_singleton
    if pipeline_singleton is None:
        pipeline_singleton = QAPipeline()
    return pipeline_singleton

