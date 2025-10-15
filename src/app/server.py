import os
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.pipeline.orchestrator import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
pipeline = None

def _get_pipeline():
	global pipeline
	if pipeline is None:
		pipeline = get_pipeline()
	return pipeline

class ChatRequest(BaseModel):
	query: str
	use_llm_rerank: Optional[bool] = None

@app.get("/health")
def health():
	return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
	pipe = _get_pipeline()
	res = await pipe.answer(
		query=req.query,
		topn=int(os.getenv("TOPN", "50")),
		topk=int(os.getenv("TOPK", "10")),
		use_llm_rerank=req.use_llm_rerank,
	)
	return res

HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>QA Bot</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #ffffff; color: #000000; }
    .wrap { display: grid; grid-template-columns: 1fr 1fr; height: 100vh; }
    .left, .right { padding: 12px; box-sizing: border-box; }
    .left { border-right: 1px solid #e0e0e0; display: flex; flex-direction: column; }
    .chat { flex: 1; overflow: auto; border: 1px solid #e0e0e0; padding: 10px; }
    .input { display: flex; gap: 8px; margin-top: 8px; }
    .input textarea { flex: 1; height: 60px; background: #f0f0f0; color: #000000; }
    .send { background: #10a37f; color: #fff; border: 0; padding: 8px 12px; }
    .ctx, .metrics { height: 50%; overflow: auto; border: 1px solid #e0e0e0; padding: 10px; background: #f9f9f9; }
  </style>
  <script>
    async function sendQuery() {
      const q = document.getElementById('q').value.trim();
      const useLLM = document.getElementById('llm').checked;
      if (!q) return;
      const chat = document.getElementById('chat');
      chat.innerHTML += `<div><b>Вы:</b> ${q}</div>`;
      document.getElementById('q').value = '';
      const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, use_llm_rerank: useLLM }) });
      const data = await res.json();
      const ans = (data.answer && data.answer.answer) || 'Нет ответа';
      chat.innerHTML += `<div><b>AI:</b> ${ans}</div>`;
      const ctxDiv = document.getElementById('ctx');
      ctxDiv.innerHTML = '';
      (data.contexts || []).forEach((c, i) => {
        const sc = (c.llm_score ?? c.score ?? 0).toFixed(3);
        const bleurt = (c.bleurt ?? 0).toFixed(3);
        const sas = (c.sas ?? 0).toFixed(3);
        ctxDiv.innerHTML += `<div style="margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;"><b>[${i+1}]</b> <span style="color: #666;">score: ${sc} | BLEURT: ${bleurt} | SAS: ${sas}</span><br/><div style="margin-top: 5px;">${c.text}</div></div>`;
      });
      const m = data.metrics || {};
      const metrics = document.getElementById('metrics');
      metrics.innerHTML = '<h4>Метрики ответа:</h4>';
      Object.keys(m).forEach(k => {
        const val = typeof m[k] === 'number' ? m[k].toFixed(3) : m[k];
        metrics.innerHTML += `<div><b>${k}:</b> ${val}</div>`;
      });
    }
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('send').addEventListener('click', sendQuery);
    });
  </script>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"left\">
      <div id=\"chat\" class=\"chat\"></div>
      <div class=\"input\">
        <textarea id=\"q\" placeholder=\"Введите вопрос...\"></textarea>
        <label><input type=\"checkbox\" id=\"llm\"/> LLM-реранкинг</label>
        <button id=\"send\" class=\"send\">Отправить</button>
      </div>
    </div>
    <div class=\"right\">
      <div id=\"ctx\" class=\"ctx\"></div>
      <div id=\"metrics\" class=\"metrics\"></div>
    </div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
	return HTML

