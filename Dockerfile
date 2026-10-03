FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PADDLE_HOME=/app/.cache/paddle \
    PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex \
    PADDLE_PDX_MODEL_SOURCE=HUGGINGFACE

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-models.txt \
    requirements-paddle-cpu.txt requirements-paddle-gpu.txt \
    requirements-torch-cpu.txt requirements-torch-gpu.txt ./

ARG INSTALL_MODELS=true
ARG MODEL_RUNTIME=gpu
ARG PADDLE_GPU_INDEX_URL
ARG TORCH_GPU_INDEX_URL
RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_MODELS" = "true" ]; then \
      if [ "$MODEL_RUNTIME" = "gpu" ]; then \
        test -n "$PADDLE_GPU_INDEX_URL" || \
          (echo "PADDLE_GPU_INDEX_URL is required for MODEL_RUNTIME=gpu" >&2; exit 1); \
        test -n "$TORCH_GPU_INDEX_URL" || \
          (echo "TORCH_GPU_INDEX_URL is required for MODEL_RUNTIME=gpu" >&2; exit 1); \
        python -m pip install -r requirements-paddle-gpu.txt \
          --index-url "$PADDLE_GPU_INDEX_URL" && \
        python -m pip install -r requirements-torch-gpu.txt \
          --index-url "$TORCH_GPU_INDEX_URL"; \
      elif [ "$MODEL_RUNTIME" = "cpu" ]; then \
        python -m pip install -r requirements-paddle-cpu.txt \
          -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
        python -m pip install -r requirements-torch-cpu.txt \
          --index-url https://download.pytorch.org/whl/cpu; \
      else \
        echo "MODEL_RUNTIME must be 'cpu' or 'gpu'" >&2; exit 1; \
      fi && \
      python -m pip install -r requirements-models.txt; \
    else \
      python -m pip install -r requirements.txt; \
    fi

COPY . .

RUN mkdir -p /app/.cache/huggingface /app/.cache/paddlex /app/data/outputs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
