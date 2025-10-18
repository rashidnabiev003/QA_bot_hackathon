import os
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import pandas as pd

from src.schemas.pydantic_schemas import BuildConfig, MetricConfig
from src.retriever.semantic_search import SemanticSearch
from src.retriever.metrics import MetricComputer
from src.llm.llm_reranker_vllm import rerank_with_llm
from src.llm.llm_chat_bot_ollama import generate_answer

load_dotenv()

logger = logging.getLogger(__name__)

class QAPipeline:
    def __init__(self, index_dir: str = None):
        if index_dir is None:
            index_dir = os.getenv("INDEX_DIR")
        self.index_dir = index_dir
        self.build_config = BuildConfig(
            batch_size=16,
            force_cpu=False,
            rerank_device=os.getenv("RERANK_DEVICE", "cpu"),
            use_fp16_rerank=True,
            chunk_size=128,
            overlap_size=50,
        )
        self.metric_config = MetricConfig(
            bleurt_ckpt=os.getenv("BLEURT_CHECKPOINT", "BLEURT-20"),
            sas_model=os.getenv("SAS_MODEL", "BAAI/bge-reranker-v2-m3"),
            sas_device=os.getenv("SAS_DEVICE", "cpu"),
            sas_fp16=False,
            bleurt_endpoint=os.getenv("BLEURT_URL", "http://localhost:8080"),
        )
        self.engine = SemanticSearch(self.index_dir, self.build_config)
        self._index_checked = False
        self.mc = MetricComputer(self.metric_config)
        self.gold_map: Dict[str, str] = {}
        self._load_gold_triplets()
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
            if not getattr(self.engine, "ready", False):
                auto = os.getenv("AUTO_BUILD_INDEX", "0") in ("1", "true", "True")
                if auto:
                    self.build_index_from_docx()
                self.engine.load()
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            auto = os.getenv("AUTO_BUILD_INDEX", "0") in ("1", "true", "True")
            if auto:
                self.build_index_from_docx()
            self.engine.load()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower()).strip()

    def _load_gold_triplets(self) -> None:
        """Загружает base_triplets.xlsx и кэширует question->gold_answer."""
        try:
            xlsx_path = os.getenv("GOLD_XLSX_PATH", str(Path("src/data/base_triplets.xlsx").resolve()))
            p = Path(xlsx_path)
            if not p.exists():
                logger.warning(f"Gold XLSX not found: {xlsx_path}")
                return
            df = pd.read_excel(p)
            df.columns = [c.lower().strip() for c in df.columns]
            if "question" not in df.columns or "answer" not in df.columns:
                logger.warning("Gold XLSX missing required columns: question, answer")
                return
            cnt = 0
            for _, row in df.iterrows():
                q = str(row.get("question") or "").strip()
                a = str(row.get("answer") or "").strip()
                if q and a:
                    self.gold_map[self._normalize(q)] = a
                    cnt += 1
            logger.info(f"Loaded gold triplets: {cnt} items from {xlsx_path}")
        except Exception as e:
            logger.error(f"Failed to load gold XLSX: {e}")

    def rebuild_index(self) -> None:
        """Удаляет старый индекс и пересобирает его из DOCX."""
        try:
            try:
                self.engine.clear_index()
            except Exception as e:
                logger.warning(f"Failed to clear index: {e}")
            self.build_index_from_docx()
            self.engine.load()
            logger.info("Index rebuilt successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")

    def build_index_from_docx(self) -> None:
        docx_path = os.getenv("DOCX_PATH", str(Path("src/data/input.docx").resolve()))
        p = Path(docx_path)
        if not p.exists():
            logger.warning(f"DOCX file not found: {docx_path}")
            return
        try:
            logger.info("Building index from DOCX via LangChain loader: %s", docx_path)
            from langchain_community.document_loaders import Docx2txtLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from transformers import AutoTokenizer

            # Загружаем документ
            loader = Docx2txtLoader(str(p))
            docs = loader.load()
            if not docs:
                logger.warning("DOCX loader returned no documents")
                return

            text = "\n\n".join(d.page_content for d in docs if d and d.page_content)
            if not text.strip():
                logger.warning("Loaded DOCX text is empty after join")
                return

            self.engine.build_from_text(text)
            logger.info("Index built from DOCX using LangChain loader")
        except Exception as e:
            logger.error(f"Failed to build index from DOCX: {e}")

    async def answer(self, query: str, topn: int = 50, topk: int = 5, use_llm_rerank: Optional[bool] = None) -> Dict[str, Any]:
        import time
        t_start = time.monotonic()
        if not self._index_checked:
            self._ensure_index_loaded()
            self._index_checked = True
        if use_llm_rerank is None:
            use_llm_rerank = os.getenv("USE_LLM_RERANK", "0")
        logger.info("Answer pipeline: topn=%s, topk=%s, llm_rerank=%s", topn, topk, use_llm_rerank)
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
                vllm_url=os.getenv("VLLM_URL"),
                model=os.getenv("VLLM_MODEL"),
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
            ollama_url=os.getenv("OLLAMA_URL"),
            model=os.getenv("OLLAMA_MODEL"),
        )
        logger.info("LLM answer generated in %.3fs", time.monotonic() - t2)
        
        # Извлекаем текст ответа
        answer_text = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
        
        metrics: Dict[str, Any] = {
            "confidence": answer.get("confidence", "unknown") if isinstance(answer, dict) else "unknown",
            "sources_used": len(answer.get("sources_used", []) if isinstance(answer, dict) else []),
            "contexts_found": len(contexts),
        }
        
        out = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "metrics": metrics,
        }
        logger.info("Answer pipeline total time: %.3fs", time.monotonic() - t_start)
        return out

    async def benchmark_from_xlsx(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        use_llm_rerank: Optional[bool] = None,
        topn: int = 50,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """Собирает бенчмарк из XLSX с триплетами (id, question, answer).
        Возвращает путь к сохранённому XLSX с метриками и контекстами."""
        xlsx_path = path or str(Path("src/data/base_triplets.xlsx").resolve())
        if not Path(xlsx_path).exists():
            return {"ok": False, "error": f"XLSX not found: {xlsx_path}"}
        
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            return {"ok": False, "error": f"Failed to read XLSX: {e}"}
        
        # Нормализуем названия колонок
        df.columns = [c.lower().strip() for c in df.columns]
        
        required = ["id", "question", "answer", "context"]
        for col in required:
            if col not in df.columns:
                return {"ok": False, "error": f"Missing column: {col}"}
        
        if limit is not None and limit > 0:
            df = df.head(int(limit))
        
        rows: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            qid = row.get("id")
            question = str(row.get("question") or "").strip()
            gold_answer = str(row.get("answer") or "").strip()
            gold_context = str(row.get("context") or "").strip()
            
            if not question:
                continue
            
            logger.info(f"Benchmark: processing question {idx+1}/{len(df)}: {question[:50]}...")
            
            resp = await self.answer(query=question, topn=topn, topk=topk, use_llm_rerank=use_llm_rerank)
            
            ans_data = resp.get("answer", {})
            if isinstance(ans_data, dict):
                answer_text = ans_data.get("answer", "")
            else:
                answer_text = str(ans_data)
            
            metrics = resp.get("metrics", {})
            contexts = resp.get("contexts", [])
            
            # Вычисляем метрики: генеративные по gold_answer, retrieval по gold_context
            try:
                metrics_result = await asyncio.to_thread(
                    self.mc.compute_all,
                    query=question,
                    answer=answer_text,
                    contexts=[c["text"] for c in contexts],
                    gold_answer=gold_answer if gold_answer else None,
                    gold_context=gold_context if gold_context else None,
                )
            except Exception as e:
                logger.error(f"Failed to compute metrics for question {qid}: {e}")
                metrics_result = {
                    "bleurt": 0,
                    "sas": 0,
                    "rouge_l": 0,
                    "ndcg": 0,
                    "mrr": 0,
                    "map": 0,
                }
            
            # Формируем запись
            rec: Dict[str, Any] = {
                "id": qid,
                "question": question,
                "answer": answer_text,
                "gold_answer": gold_answer,
                "gold_context": gold_context,
                # Метрики генерации
                "bleurt": metrics_result.get("bleurt", 0),
                "sas": metrics_result.get("sas", 0),
                "rouge_l": metrics_result.get("rouge_l", 0),
                # Retrieval метрики
                "ndcg": metrics_result.get("ndcg", 0),
                "mrr": metrics_result.get("mrr", 0),
                "map": metrics_result.get("map", 0),
                # Общие метрики из answer
                "confidence": metrics.get("confidence", "unknown"),
                "sources_used": metrics.get("sources_used", 0),
                "contexts_found": metrics.get("contexts_found", 0),
            }
            
            for i, ctx in enumerate(contexts):
                rec[f"context_{i+1}"] = ctx.get("text", "")
                if use_llm_rerank and "llm_score" in ctx:
                    rec[f"context_{i+1}_llm_score"] = ctx.get("llm_score", 0.0)
                elif "score_rerank" in ctx:
                    rec[f"context_{i+1}_score_rerank"] = ctx.get("score_rerank", 0.0)
                elif "score" in ctx:
                    rec[f"context_{i+1}_score"] = ctx.get("score", 0.0)
            
            rows.append(rec)
        
        if not rows:
            return {"ok": False, "error": "No rows processed"}
        
        # Сохраняем результат
        out_df = pd.DataFrame(rows)
        out_path = str(Path("src/data/benchmark_results.xlsx").resolve())
        out_df.to_excel(out_path, index=False)
        
        logger.info(f"Benchmark completed: {len(out_df)} rows saved to {out_path}")
        return {"ok": True, "path": out_path, "rows": len(out_df)}

pipeline_singleton: Optional[QAPipeline] = None

def get_pipeline() -> QAPipeline:
    global pipeline_singleton
    if pipeline_singleton is None:
        pipeline_singleton = QAPipeline()
    return pipeline_singleton

