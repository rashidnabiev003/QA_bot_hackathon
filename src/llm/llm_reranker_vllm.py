import asyncio
import json
import logging
import os
import sys
from functools import lru_cache
from typing import Any

import aiohttp

logging.basicConfig(
	level=logging.INFO,
	format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
	handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _read_prompt() -> tuple[str, str]:
	"""Кэшированное чтение промптов из файлов"""
	base = os.path.join(
		os.path.dirname(__file__), '..', 'prompts', f'llm_reranking'
	)
	system_path = os.path.abspath(os.path.join(base, 'system.txt'))
	user_path = os.path.abspath(os.path.join(base, 'user.txt'))
	with open(system_path, encoding='utf-8') as f:
		system_prompt = f.read()
	with open(user_path, encoding='utf-8') as f:
		user_prompt = f.read()
	return system_prompt, user_prompt


def _safe_fill(template: str, mapping: dict[str, str]) -> str:
	out = template
	for k, v in mapping.items():
		out = out.replace('{' + k + '}', v)
	return out


def _extract_score_from_text(text: str) -> float:
	try:
		import re

		json_match = re.search(r'\{.*\}', text, re.DOTALL)
		if json_match:
			data = json.loads(json_match.group(0))
			v = data.get('score')
			if isinstance(v, (int, float)):
				return float(v)
		# первое число с плавающей точкой 0.x или 1.0
		num_match = re.search(r'(?:(?:0(?:\.\d+)?|1(?:\.0+)?))', text)
		if num_match:
			return float(num_match.group(0))
		return 0.0
	except Exception as e:
		logger.warning(f'Failed to extract score: {e}')
		return 0.0


async def _score_one(
	session: aiohttp.ClientSession,
	vllm_url: str,
	model: str,
	system_prompt: str,
	user_prompt: str,
	query: str,
	chunk_text: str,
) -> float:
	chunk_text = (chunk_text or '')[:500]
	user_message = _safe_fill(
		user_prompt, {'query': query, 'chunk': chunk_text}
	)
	messages = [
		{'role': 'system', 'content': system_prompt},
		{'role': 'user', 'content': user_message},
	]
	payload = {
		'model': model,
		'messages': messages,
		'temperature': 0.0,
		'max_tokens': 16,
		'response_format': {'type': 'json_object'},
	}
	try:
		async with session.post(
			f'{vllm_url}/v1/chat/completions',
			json=payload,
			timeout=aiohttp.ClientTimeout(total=60),
		) as resp:
			if resp.status != 200:
				error_text = await resp.text()
				logger.warning(
					f'vLLM returned {resp.status}: {error_text[:200]}'
				)
				return 0.0
			data = await resp.json()
			text = (
				data.get('choices', [{}])[0]
				.get('message', {})
				.get('content', '')
			)
			s = _extract_score_from_text(text)
			if s is None:
				logger.warning(
					f'No score extracted from vLLM response: {text[:120]}'
				)
				return 0.0
			if s > 1.0:
				s = s / 10.0
			if s < 0.0:
				s = 0.0
			if s > 1.0:
				s = 1.0
			return s
	except Exception as e:
		logger.warning(f'Error scoring chunk: {e}')
		return 0.0


async def rerank_with_llm(
	query: str,
	chunks: list[dict[str, Any]],
	topk: int = 5,
	vllm_url: str = 'http://localhost:8000',
	model: str = 'Qwen/Qwen3-4B-Thinking-2507',
) -> list[dict[str, Any]]:
	"""Асинхронный LLM-реранкинг"""
	logger.info(
		f'Starting LLM reranking for {len(chunks)} chunks with model {model} at {vllm_url}'
	)
	system_prompt, user_prompt = _read_prompt()
	try:
		async with aiohttp.ClientSession() as session:
			tasks = []
			for ch in chunks:
				chunk_text = ch.get('text', '')
				tasks.append(
					_score_one(
						session,
						vllm_url,
						model,
						system_prompt,
						user_prompt,
						query,
						chunk_text,
					)
				)
			logger.info(
				f'Created {len(tasks)} per-chunk scoring tasks, starting parallel execution...'
			)
			scores = await asyncio.gather(*tasks, return_exceptions=False)
			logger.info(f'Received scores: {scores[:5]}... (showing first 5)')
			for i, ch in enumerate(chunks):
				ch['llm_score'] = float(scores[i])
			reranked = sorted(
				chunks, key=lambda x: x.get('llm_score', 0.0), reverse=True
			)
			logger.info(
				f'Reranked {len(reranked)} chunks, returning top {topk}'
			)
			return reranked[:topk]
	except Exception as e:
		logger.error(f'Error in async reranking: {e}', exc_info=True)
		return chunks[:topk]
