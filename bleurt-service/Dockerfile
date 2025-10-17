FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLEURT_CHECKPOINT=/models/BLEURT-20

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /srс

RUN pip install --upgrade pip setuptools wheel && \
    pip install fastapi==0.115.5 uvicorn==0.30.6 \
      "tensorflow-cpu==2.18.1" \
      git+https://github.com/google-research/bleurt@master

COPY app.py ./app.py
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
