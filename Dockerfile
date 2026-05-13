# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv for fast installs
RUN pip install --no-cache-dir uv==0.4.27

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./
RUN uv pip install --system -e ".[dev]"

# scispaCy model
RUN pip install --no-cache-dir https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY ui/ ./ui/

# Create data dirs (will be mounted via volume in production)
RUN mkdir -p data/{trec_ct,ctgov_snapshot,mesh,indices} .cache

EXPOSE 8000

CMD ["uvicorn", "trial_matcher.api:app", "--host", "0.0.0.0", "--port", "8000"]
