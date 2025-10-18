from __future__ import annotations
import urllib.request
import json, re, os, logging
from typing import Optional, List, Dict, Any, Union
import numpy as np
import math

from nltk.stem.snowball import RussianStemmer
from FlagEmbedding import FlagReranker
from src.schemas.pydantic_schemas import MetricConfig

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
        target_device = "cpu" 
        self.sas = FlagReranker(cfg.sas_model, use_fp16=cfg.sas_fp16, device=target_device)

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
        try:
            scores = self.sas.compute_score([[pred, ref]], normalize=True)
            if isinstance(scores, list) and len(scores) > 0:
                return float(scores[0])
            elif isinstance(scores, (int, float)):
                return float(scores)
            return 0.0
        except Exception as e:
            logging.warning(f"SAS compute_score failed: {e}")
            return 0.0

    def rougeL_ru(self, pred: str, ref: str) -> float:
        return float(rouge_l_f1_ru(pred, ref))

    def retrieval_metrics_semantic(self, gold_ctx: Union[str, List[str]], retrieved_texts: List[str], k: int = 10, rel_threshold: float = 0.35) -> Dict[str, float]:
        if not retrieved_texts:
            return {"ndcg": 0.0, "mrr": 0.0, "map": 0.0}

        gold_list: List[str] = [gold_ctx] if isinstance(gold_ctx, str) else list(gold_ctx)
        sims: List[float] = []
        for t in retrieved_texts[:k]:
            pair_scores = []
            for g in gold_list:
                try:
                    s = self.sas.compute_score([[t, g]], normalize=True)
                    s = float(s[0] if isinstance(s, list) else s)
                except Exception:
                    s = 0.0
                pair_scores.append(s)
            sims.append(max(pair_scores) if pair_scores else 0.0)

        rel = [1 if s >= rel_threshold else 0 for s in sims]

        try:
            rank_first = rel.index(1)
            mrr = 1.0 / (rank_first + 1)
        except ValueError:
            mrr = 0.0
        
        gains = sims
        dcg  = sum(gains[i] / math.log2(i + 2) for i in range(len(gains)))
        ideal = sorted(gains, reverse=True)
        idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
                
        relevant_positions = [i+1 for i, r in enumerate(rel) if r == 1]
        if relevant_positions:
            precisions = [(j+1) / pos for j, pos in enumerate(relevant_positions)]
            ap = sum(precisions) / len(precisions)
        else:
            ap = 0.0

        return {"ndcg": float(ndcg), "mrr": float(mrr), "map": float(ap)}
    
    def compute_all(self, query: str, answer: str, contexts: List[str], gold_answer: Optional[str] = None, gold_context: Optional[Union[str, List[str]]] = None,) -> Dict[str, Any]:
        result = {
            "bleurt": 0.0,
            "sas": 0.0,
            "rouge_l": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "map": 0.0,
        }

        if gold_answer:
            result["bleurt"]  = self.bleurt20(answer, gold_answer)
            result["sas"]     = self.sas_user_bge_m3(answer, gold_answer)
            result["rouge_l"] = self.rougeL_ru(answer, gold_answer)

        if gold_context and contexts:
            k = min(5, len(contexts))
            retr = self.retrieval_metrics_semantic(gold_context, contexts, k=k, rel_threshold=0.75)
            result["ndcg"] = retr["ndcg"]
            result["mrr"]  = retr["mrr"]
            result["map"]  = retr["map"]

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
        out["retrieval"] = mc.retrieval_metrics_semantic(gold_context, texts, k=10)
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