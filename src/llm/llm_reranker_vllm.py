import os
import sys
import json
import logging
import aiohttp
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(
	level=logging.INFO,
	format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
	handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=4)
def _read_prompt(stage: int) -> Tuple[str, str]:
	"""Кэшированное чтение промптов из файлов"""
	base = os.path.join(os.path.dirname(__file__), "..", "prompts", f"llm_reranking")
	system_path = os.path.abspath(os.path.join(base, "system.txt"))
	user_path = os.path.abspath(os.path.join(base, "user.txt"))
	with open(system_path, "r", encoding="utf-8") as f:
		system_prompt = f.read()
	with open(user_path, "r", encoding="utf-8") as f:
		user_prompt = f.read()
	return system_prompt, user_prompt

def _extract_scores_from_json(text: str) -> List[float]:
	"""Извлекает список оценок из JSON ответа"""
	try:
		import re
		json_match = re.search(r'\{.*\}', text, re.DOTALL)
		if json_match:
			data = json.loads(json_match.group(0))
			if "scores" in data:
				return [float(s) for s in data["scores"]]
		return []
	except Exception as e:
		logger.warning(f"Failed to extract scores: {e}")
		return []

async def rerank_with_llm(
	query: str,
	chunks: List[Dict[str, Any]],
	topk: int = 5,
	vllm_url: str = "http://localhost:8000",
	model: str = "Qwen/Qwen3-4B-Thinking-2507"
) -> List[Dict[str, Any]]:
	"""Реранкинг чанков с использованием vLLM"""
	system_prompt, user_prompt = _read_prompt(1)
	
	chunks_text = "\n".join([
		f"{i+1}. {chunk['text'][:200]}..."
		for i, chunk in enumerate(chunks)
	])
	
	user_message = user_prompt.format(query=query, chunks=chunks_text)
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_message}
	]
	
	payload = {
		"model": model,
		"messages": messages,
		"temperature": 0.1,
		"max_tokens": 512
	}
	
	try:
		async with aiohttp.ClientSession() as session:
			async with session.post(
				f"{vllm_url}/v1/chat/completions",
				json=payload,
				timeout=aiohttp.ClientTimeout(total=60)
			) as resp:
				if resp.status != 200:
					error_text = await resp.text()
					logger.error(f"vLLM error: {resp.status} - {error_text}")
					return chunks[:topk]
				
				result = await resp.json()
				response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
				scores = _extract_scores_from_json(response_text)
				
				if len(scores) != len(chunks):
					logger.warning(f"Scores count mismatch: {len(scores)} vs {len(chunks)}")
					return chunks[:topk]
				
				for i, chunk in enumerate(chunks):
					chunk["llm_score"] = scores[i]
				
				reranked = sorted(chunks, key=lambda x: x.get("llm_score", 0), reverse=True)
				logger.info(f"Reranked {len(chunks)} chunks, returning top {topk}")
				
				return reranked[:topk]
				
	except Exception as e:
		logger.error(f"Error calling vLLM: {e}", exc_info=True)
		return chunks[:topk]