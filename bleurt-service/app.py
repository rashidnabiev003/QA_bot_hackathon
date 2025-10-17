import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import bleurt.score as bleurt_score
except Exception:  # pragma: no cover
    bleurt_score = None

BLEURT_CKPT = os.getenv("BLEURT_CHECKPOINT", "BLEURT-20")

app = FastAPI()
scorer = None

class ScoreRequest(BaseModel):
    references: List[str]
    candidates: List[str]

@app.on_event("startup")
def _load_model():
    global scorer
    if bleurt_score is None:
        return
    try:
        scorer = bleurt_score.BleurtScorer(BLEURT_CKPT)
    except Exception:
        scorer = None

@app.get("/health")
def healthz():
    return {"ok": scorer is not None}

@app.post("/score")
def score_endpoint(req: ScoreRequest):
    if scorer is None:
        raise HTTPException(status_code=503, detail="bleurt not loaded")
    if len(req.references) != len(req.candidates):
        raise HTTPException(status_code=400, detail="length mismatch")
    scores = scorer.score(references=req.references, candidates=req.candidates)
    return {"scores": [float(s) for s in scores]}


