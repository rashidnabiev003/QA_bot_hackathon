FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /srv

RUN pip install --upgrade pip setuptools wheel && \
	pip install fastapi==0.115.5 uvicorn==0.30.6 bleurt==0.0.2 numpy==2.1.3

COPY app.py ./app.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    BLEURT_CHECKPOINT=/models/BLEURT-20

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn[standard]==0.30.0 \
    "tensorflow-cpu==2.17.*" "tensorflow-text==2.17.*" \
    git+https://github.com/google-research/bleurt@master

WORKDIR /app
COPY app /app

EXPOSE 8080
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080","--workers","1"]
