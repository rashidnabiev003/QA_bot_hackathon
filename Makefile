SHELL := /usr/bin/zsh

APP_IMAGE := qa-bot-app
BLEURT_IMAGE := bleurt-service:cpu

.PHONY: help build up down logs app-shell index docx

help:
	@echo "Targets: build up down logs app-shell index docx"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

app-shell:
	docker exec -it qa_bot_app /bin/sh

# Индексация текста из файла (plain text)
index:
	python - <<'PY'
from src.retriever.semantic_search import SemanticSearch
from src.schemas.pydantic_schemas import BuildConfig
import sys, os
text_path = os.environ.get('TEXT', 'data/input.txt')
with open(text_path, 'r', encoding='utf-8') as f:
	text = f.read()
cfg = BuildConfig(batch_size=32, force_cpu=False, rerank_device='cuda', use_fp16_rerank=True, chunk_size=500, overlap_size=50)
engine = SemanticSearch('./index', cfg)
engine.build_from_text(text)
print('Index built: chunks', len(engine.meta))
PY

# Извлечение текста из DOCX и индексация
docx:
	python - <<'PY'
from src.utils.file_loader import parse_docx_to_text
from src.retriever.semantic_search import SemanticSearch
from src.schemas.pydantic_schemas import BuildConfig
import sys, os
docx_path = os.environ.get('DOCX', 'data/input.docx')
text = parse_docx_to_text(docx_path)
cfg = BuildConfig(batch_size=32, force_cpu=False, rerank_device='cuda', use_fp16_rerank=True, chunk_size=500, overlap_size=50)
engine = SemanticSearch('./index', cfg)
engine.build_from_text(text)
print('Index built from DOCX: chunks', len(engine.meta))
PY

