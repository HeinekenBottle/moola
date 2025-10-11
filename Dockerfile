# Dockerfile supports CPU by default. For CUDA, change base image accordingly.
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /workspace/moola
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .
COPY . .

# Default command runs the local pipeline; override in RunPod/Erden.
CMD ["bash", "scripts/run_local.sh"]
