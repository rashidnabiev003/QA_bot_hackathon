FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip setuptools wheel && \
	pip install \
	"numpy>=2.0.0" \
	"pydantic==2.11.10" \
	"sentence-transformers>=5.0.0" \
	"transformers>=4.53.1" \
	"torch==2.7.0" \
	"torchaudio>=2.7.0" \
	"torchvision>=0.22.0" \
	"nltk>=3.8.0" \
	"aiohttp>=3.9.0" \
	"faiss-cpu>=1.8.0" \
	"FlagEmbedding>=1.2.0" \
	"fastapi>=0.115.0" \
	"uvicorn>=0.30.0" \
	"python-dotenv>=1.0.0"

COPY src ./src
COPY README.md ./

RUN python -c "import nltk; nltk.download('punkt')"

EXPOSE 8000

CMD ["python", "-m", "src.app.front"]
