import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.app.static.index import get_html
from src.pipeline.orchestrator import get_pipeline
from src.schemas.pydantic_schemas import BenchmarkRequest, ChatRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
os.environ.setdefault(
	'CUDA_VISIBLE_DEVICES', os.getenv('CUDA_VISIBLE_DEVICES', '0')
)
logger = logging.getLogger(__name__)

app = FastAPI()
pipeline = None


def _get_pipeline():
	global pipeline
	if pipeline is None:
		pipeline = get_pipeline()
	return pipeline


@app.get('/health')
def health():
	return {'ok': True}


@app.post('/chat')
async def chat(req: ChatRequest):
	pipe = _get_pipeline()
	return await pipe.answer(
		query=req.query,
		topn=int(os.getenv('TOPN', '50')),
		topk=int(os.getenv('TOPK', '5')),
		use_llm_rerank=req.use_llm_rerank,
	)


@app.post('/rebuild_index')
async def rebuild_index():
	try:
		pipe = _get_pipeline()
		await asyncio.to_thread(pipe.rebuild_index)
		return {'ok': True, 'message': 'Index rebuilt successfully'}
	except Exception as e:
		logger.exception('Rebuild index failed')
		return {'ok': False, 'error': str(e)}


@app.get('/', response_class=HTMLResponse)
def index():
	return get_html()


@app.post('/benchmark')
async def benchmark(req: BenchmarkRequest):
	pipe = _get_pipeline()
	return await pipe.benchmark_from_xlsx(
		path=req.path,
		limit=req.limit,
		use_llm_rerank=req.use_llm_rerank,
		topn=int(os.getenv('TOPN', '50')),
		topk=int(os.getenv('TOPK', '5')),
	)
