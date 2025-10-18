FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
	ln -s /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml ./
COPY uv.lock ./
COPY .env ./.env
COPY README.md ./
RUN uv sync --no-install-project --no-dev --frozen
COPY src ./src

RUN . .venv/bin/activate && python -c "import nltk; nltk.download('punkt')"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.app.server:app", "--host", "0.0.0.0", "--port", "8000"]
