# QA Bot с RAG

Система вопросов и ответов с использованием RAG (Retrieval-Augmented Generation) для точных ответов на основе базы знаний.

## Возможности

- 🔍 **Семантический поиск** - поиск релевантных контекстов через FAISS и BGE-M3
- 🤖 **LLM генерация ответов** - использование Ollama для генерации точных ответов
- 🎯 **Двойной реранкинг** - выбор между cross-encoder и LLM-based реранкингом
- 📊 **Метрики качества** - BLEURT, SAS, ROUGE-L для оценки ответов
- 🖥️ **Удобный интерфейс** - Tkinter GUI с 3 панелями

## Архитектура

### Компоненты

1. **Semantic Search** (`src/retriever/semantic_search.py`)
   - Индексация документов через FAISS
   - Эмбеддинги через BGE-M3
   - Первичный поиск топ-N документов

2. **LLM Reranker** (`src/llm/llm_reranker_vllm.py`)
   - Переранжирование через vLLM
   - Опциональная замена cross-encoder

3. **Chat Bot** (`src/llm/llm_chat_bot_ollama.py`)
   - Генерация ответов через Ollama
   - Использование контекста из retriever

4. **Metrics** (`src/retriever/metrics.py`)
   - BLEURT-20 для семантического сходства
   - SAS (Semantic Answer Similarity)
   - ROUGE-L с русским стеммингом
   - nDCG, MRR, MAP для retrieval метрик

## Быстрый старт

### Через Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Загрузка модели в Ollama
docker exec -it qa_bot_ollama ollama pull llama3.2:3b

# Приложение доступно на http://localhost:8000
```

### Локально

```bash
# Установка зависимостей
pip install -e .

# Запуск приложения
python run_app.py
```

## Структура интерфейса

### Левая панель - Чат
- Поле ввода вопросов
- История диалога с AI

### Правая панель (верх) - Контексты
- Топ-5 найденных фрагментов
- Оценки релевантности
- Переключатель LLM реранкинга

### Правая панель (низ) - Метрики
- Уровень уверенности
- Количество использованных источников
- Retrieval метрики

## Конфигурация

### BuildConfig
```python
BuildConfig(
    batch_size=32,           # Размер батча для эмбеддингов
    force_cpu=False,         # Принудительно CPU
    rerank_device="cuda",    # Устройство для реранкинга
    use_fp16_rerank=True,    # FP16 для реранкинга
    chunk_size=500,          # Размер чанка
    overlap_size=50          # Перекрытие чанков
)
```

### MetricConfig
```python
MetricConfig(
    bleurt_ckpt="BLEURT-20",
    sas_model="BAAI/bge-m3",
    sas_device="cuda",
    sas_fp16=True
)
```

## Индексация документов

```python
from src.retriever.semantic_search import SemanticSearch
from src.schemas.pydantic_schemas import BuildConfig

config = BuildConfig(
    batch_size=32,
    force_cpu=False,
    rerank_device="cuda",
    use_fp16_rerank=True,
    chunk_size=500,
    overlap_size=50
)

search = SemanticSearch("./index", config)

# Индексация текста
with open("your_document.txt", "r", encoding="utf-8") as f:
    text = f.read()
    search.build_from_text(text)
```

## Зависимости

- Python 3.12
- PyTorch 2.7.0
- FAISS для индексации
- FlagEmbedding (BGE-M3)
- Ollama для LLM
- vLLM для реранкинга
- BLEURT для метрик

## Сервисы

- **app** (порт 8000) - основное приложение
- **ollama** (порт 11434) - LLM сервер
- **vllm** (порт 8001) - реранкинг сервер

## Авторы

- Набиев Рашидхон (nabievrasheed@yandex.ru)
- Никитин Борис

## Лицензия

См. файл LICENSE
