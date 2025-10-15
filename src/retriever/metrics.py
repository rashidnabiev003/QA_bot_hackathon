from __future__ import annotations
import urllib.request
import json, re, os, logging
from typing import Optional, List, Dict, Any
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from nltk.stem.snowball import RussianStemmer
from src.schemas.pydantic_schemas import MetricConfig

bleurt_score = None

def _tok_ru(x: str) -> List[str]:
    return re.findall(r"\w+", x.lower(), flags=re.UNICODE)

def _stem_ru(tokens: List[str]) -> List[str]:
    st = RussianStemmer()
    return [st.stem(t) for t in tokens]

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] > dp[j-1] else dp[j-1]
            prev = tmp
    return dp[m]

def rouge_l_f1_ru(pred: str, ref: str) -> float:
    a = _stem_ru(_tok_ru(pred))
    b = _stem_ru(_tok_ru(ref))
    if not a or not b:
        return 0.0
    l = _lcs_len(a, b)
    p = l / max(1, len(a))
    r = l / max(1, len(b))
    if p == 0.0 or r == 0.0:
        return 0.0
    return 2*p*r/(p+r)

def ndcg_at_k(rank: int, k: int = 10) -> float:
    if rank < 0 or rank >= k: return 0.0
    return 1.0/np.log2(rank+2)

def mrr_at_k(rank: int, k: int = 10) -> float:
    if rank < 0 or rank >= k: return 0.0
    return 1.0/(rank+1)

def map_at_100(rank: int) -> float:
    if rank < 0 or rank >= 100: return 0.0
    return 1.0/(rank+1)

class MetricComputer:
    def __init__(self, cfg: MetricConfig):
        self.cfg = cfg
        self.bleurt_url = cfg.bleurt_endpoint or os.getenv("BLEURT_URL")
        self.bleurt = None
        self.sas = BGEM3FlagModel(cfg.sas_model, device=cfg.sas_device, use_fp16=cfg.sas_fp16)

    def _bleurt20_http(self, pred: str, ref: str) -> float:
        try:
            payload = json.dumps({"references": [ref], "candidates": [pred]}).encode("utf-8")
            req = urllib.request.Request(
                url=f"{self.bleurt_url}/score",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                scores = data.get("scores") or []
                if scores:
                    return float(scores[0])
        except Exception as e:
            logging.warning(f"BLEURT HTTP call failed: {e}")
            return 0.0
        return 0.0

    def bleurt20(self, pred: str, ref: str) -> float:
        if self.bleurt_url:
            return self._bleurt20_http(pred, ref)
        if self.bleurt is not None:
            try:
                return float(self.bleurt.score(references=[ref], candidates=[pred])[0])
            except Exception:
                return 0.0
        return 0.0

    def sas_user_bge_m3(self, pred: str, ref: str) -> float:
        out = self.sas.compute_score([[pred, ref]])
        for k in ("colbert+sparse+dense", "sparse+dense", "colbert"):
            if isinstance(out, dict) and k in out:
                v = out[k]
                return float(v[0] if isinstance(v, list) else v)
        if isinstance(out, list): return float(out[0])
        if isinstance(out, dict) and len(out) == 1:
            v = list(out.values())[0]
            return float(v[0] if isinstance(v, list) else v)
        return 0.0

    def rougeL_ru(self, pred: str, ref: str) -> float:
        return float(rouge_l_f1_ru(pred, ref))

    def retrieval_metrics(self, gold_ctx: str, retrieved_texts: List[str], k: int = 10) -> Dict[str, float]:
        g = set(_tok_ru(gold_ctx))
        best = -1
        for i, t in enumerate(retrieved_texts):
            if len(g & set(_tok_ru(t))) > 0:
                best = i
                break
        return {
            f"ndcg@{k}": ndcg_at_k(best, k),
            f"mrr@{k}": mrr_at_k(best, k),
            "map@100": map_at_100(best),
            "best_rank": float(best),
        }
    
    def compute_all(self, query: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Вычисляет все метрики для ответа и контекстов"""
        result = {
            "bleurt_avg": 0,
            "bleurt_per_context": [],
            "sas_avg": 0,
            "sas_per_context": [],
            "rouge_l": 0,
            "ndcg": 0,
            "mrr": 0,
            "map": 0,
        }
        
        # Метрики для каждого контекста относительно ответа (ВСЕГДА вычисляем)
        for ctx in contexts:
            result["bleurt_per_context"].append(self.bleurt20(ctx, answer))
            result["sas_per_context"].append(self.sas_user_bge_m3(ctx, answer))
        
        # Средние значения
        if result["bleurt_per_context"]:
            result["bleurt_avg"] = sum(result["bleurt_per_context"]) / len(result["bleurt_per_context"])
        if result["sas_per_context"]:
            result["sas_avg"] = sum(result["sas_per_context"]) / len(result["sas_per_context"])
        
        # Метрики answer vs ground_truth (если есть)
        if ground_truth:
            result["rouge_l"] = self.rougeL_ru(answer, ground_truth)
            # Retrieval метрики
            if contexts:
                retr_metrics = self.retrieval_metrics(ground_truth, contexts, k=min(10, len(contexts)))
                result["ndcg"] = retr_metrics.get("ndcg@10", 0)
                result["mrr"] = retr_metrics.get("mrr@10", 0)
                result["map"] = retr_metrics.get("map@100", 0)
        
        return result

def evaluate_query(engine, query: str, gold_answer: str, pred_answer: str, gold_context: Optional[str] = None, topn: int = 50, topk: int = 10, mc: Optional[MetricComputer] = None) -> Dict[str, Any]:
    r = engine.retrieve(query, topn=topn, topk=topk)
    texts = [x["text"] for x in r["results"]]
    out = {
        "query": query,
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "bleurt20": (mc.bleurt20(pred_answer, gold_answer) if mc else 0.0),
        "sas_user_bge_m3": mc.sas_user_bge_m3(pred_answer, gold_answer),
        "rougeL_stem_ru_f1": mc.rougeL_ru(pred_answer, gold_answer),
        "retrieval": None,
        "topk": r["results"],
    }
    if gold_context is not None:
        out["retrieval"] = mc.retrieval_metrics(gold_context, texts, k=10)
    return out

def evaluate_many(engine, batch: List[Dict[str, Any]], mc: MetricComputer, topn: int = 50, topk: int = 10) -> List[Dict[str, Any]]:
    res = []
    for item in batch:
        res.append(evaluate_query(
            engine=engine,
            query=item["query"],
            gold_answer=item["gold_answer"],
            pred_answer=item["pred_answer"],
            gold_context=item.get("gold_context"),
            topn=topn,
            topk=topk,
            mc=mc
        ))
    return res