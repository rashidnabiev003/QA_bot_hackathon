import os
import json
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from bleurt import score as bleurt_score
except Exception as e:  # pragma: no cover
    bleurt_score = None

BLEURT_CKPT = os.getenv("BLEURT_CHECKPOINT", "BLEURT-20")

app = FastAPI()
scorer = None

@app.on_event("startup")
def _load_model():
    global scorer
    if bleurt_score is None:
        return
    try:
        scorer = bleurt_score.BleurtScorer(BLEURT_CKPT)
    except Exception:
        scorer = None

class ScoreRequest(BaseModel):
    references: List[str]
    candidates: List[str]

@app.get("/healthz")
def healthz():
    return {"ok": scorer is not None}

@app.post("/score")
def score_endpoint(req: ScoreRequest):
    if scorer is None:
        return {"scores": [0.0 for _ in req.candidates]}
    try:
        scores = scorer.score(references=req.references, candidates=req.candidates)
        return {"scores": [float(s) for s in scores]}
    except Exception:
        return {"scores": [0.0 for _ in req.candidates]}

