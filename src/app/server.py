import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.pipeline.orchestrator import get_pipeline
from src.schemas.pydantic_schemas import ChatRequest
from src.app.static.index import get_html
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
# Ограничим видимые GPU для приложения (оставляем GPU 1 под vLLM, а приложению — 0)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
logger = logging.getLogger(__name__)

app = FastAPI()
pipeline = None

def _get_pipeline():
	global pipeline
	if pipeline is None:
		pipeline = get_pipeline()
	return pipeline

@app.get("/health")
def health():
	return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
	pipe = _get_pipeline()
	res = await pipe.answer(
		query=req.query,
		topn=int(os.getenv("TOPN")),
		topk=int(os.getenv("TOPK")),
		use_llm_rerank=req.use_llm_rerank,
	)
	return res

@app.post("/rebuild_index")
async def rebuild_index():
	try:
		pipe = _get_pipeline()
		await asyncio.to_thread(pipe.rebuild_index)
		return {"ok": True, "message": "Index rebuilt successfully"}
	except Exception as e:
		logger.exception("Rebuild index failed")
		return {"ok": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index():
	return get_html()

