FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (lightgbm needs libgomp; parquet/pyarrow needs nothing extra here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Data + model output dirs
RUN mkdir -p data/raw data/processed models

# Default: run the full pipeline end-to-end
CMD ["python", "main.py"]
