import os
import re
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
	base = os.path.join(os.path.dirname(__file__), "..", "prompts", f"llm_response")
	system_path = os.path.abspath(os.path.join(base, "system.txt"))
	user_path = os.path.abspath(os.path.join(base, "user.txt"))
	with open(system_path, "r", encoding="utf-8") as f:
		system_prompt = f.read()
	with open(user_path, "r", encoding="utf-8") as f:
		user_prompt = f.read()
	return system_prompt, user_prompt

def _extract_json_from_text(text: str) -> str:
	"""Извлекает JSON из текста, удаляя markdown блоки и лишний текст"""
	
	json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
	if json_block:
		return json_block.group(1)

	json_match = re.search(r'\{.*\}', text, re.DOTALL)
	if json_match:
		return json_match.group(0)
	
	return text

def _normalize_json_string(text: str) -> str:
	"""Пытается получить валидный JSON-объект в виде строки"""
	try:
		candidate = _extract_json_from_text(text)
		obj = json.loads(candidate)
		return json.dumps(obj, ensure_ascii=False)
	except Exception:
		logger.warning("Failed to normalize JSON string", exc_info=True)
		return "{}"

async def generate_answer(
	query: str,
	contexts: List[Dict[str, Any]],
	ollama_url: str = "http://localhost:11434",
	model: str = "open-ai/gpt-oss-20b"
) -> Dict[str, Any]:
	"""Генерирует ответ на вопрос используя Ollama"""
	system_prompt, user_prompt = _read_prompt(1)
	
	context_text = "\n\n".join([
		f"Фрагмент {i+1} (score: {ctx.get('score', 0):.3f}):\n{ctx['text']}"
		for i, ctx in enumerate(contexts)
	])
	
	user_message = user_prompt.format(query=query, context=context_text)
	
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_message}
	]
	
	payload = {
		"model": model,
		"messages": messages,
		"stream": False,
		"format": "json"
	}
	
	try:
		async with aiohttp.ClientSession() as session:
			async with session.post(
				f"{ollama_url}/api/chat",
				json=payload,
				timeout=aiohttp.ClientTimeout(total=60)
			) as resp:
				if resp.status != 200:
					error_text = await resp.text()
					logger.error(f"Ollama error: {resp.status} - {error_text}")
					return {
						"answer": "Ошибка при генерации ответа",
						"confidence": "low",
						"sources_used": []
					}
				
				result = await resp.json()
				response_text = result.get("message", {}).get("content", "{}")
				normalized = _normalize_json_string(response_text)
				response_data = json.loads(normalized)
				
				logger.info(f"Generated answer for query: {query[:50]}...")
				return response_data
				
	except Exception as e:
		logger.error(f"Error calling Ollama: {e}", exc_info=True)
		return {
			"answer": "Произошла ошибка при генерации ответа",
			"confidence": "low",
			"sources_used": []
		}