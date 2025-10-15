import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from bleurt import score as bleurt_score

class ScoreIn(BaseModel):
    candidates: List[str]
    references: List[str]

    @field_validator("references")
    @classmethod
    def _same_len(cls, v, info):
        c = info.data.get("candidates", [])
        if len(v) != len(c):
            raise ValueError("length mismatch")
        return v

class ScoreOut(BaseModel):
    scores: List[float]

app = FastAPI()
_scorer = None

def _get_scorer():
    global _scorer
    if _scorer is None:
        ckpt = os.getenv("BLEURT_CHECKPOINT", "/models/BLEURT-20")
        _scorer = bleurt_score.BleurtScorer(ckpt)
    return _scorer

@app.get("/healthz")
def healthz():
    try:
        _ = _get_scorer()
        return {"ok": True}
    except Exception:
        raise HTTPException(status_code=500, detail="bleurt not ready")

@app.post("/score", response_model=ScoreOut)
def score(inp: ScoreIn):
    s = _get_scorer()
    vals = s.score(references=inp.references, candidates=inp.candidates)
    return ScoreOut(scores=[float(x) for x in vals])
