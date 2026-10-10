FROM python:3.11-slim

ARG MODEL_RUNTIME=gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PADDLE_HOME=/app/.cache/paddle \
    PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex \
    PADDLE_PDX_MODEL_SOURCE=HUGGINGFACE \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_INDEX_URL=https://pypi.iranserver.com/repository/pypi/simple \
    PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST="pypi.iranserver.com mirrors.aliyun.com"


WORKDIR /app

# Prefer IranServer; fall back to Aliyun when a package/version is missing (404).
RUN printf '%s\n' \
      '[global]' \
      'index-url = https://pypi.iranserver.com/repository/pypi/simple' \
      'extra-index-url = https://mirrors.aliyun.com/pypi/simple' \
      'trusted-host = pypi.iranserver.com mirrors.aliyun.com' \
      'timeout = 60' \
      > /etc/pip.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-models.txt \
    requirements-paddle-cpu.txt requirements-paddle-gpu.txt \
    requirements-torch-cpu.txt requirements-torch-gpu.txt ./

RUN set -eux; \
    python -m pip install --upgrade pip; \
    case "$MODEL_RUNTIME" in \
      gpu) \
        python -m pip install -r requirements-paddle-gpu.txt \
          --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/; \
        python -m pip install -r requirements-torch-gpu.txt \
          --index-url https://download.pytorch.org/whl/cu130; \
        ;; \
      cpu) \
        python -m pip install -r requirements-paddle-cpu.txt \
          --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/; \
        python -m pip install -r requirements-torch-cpu.txt \
          --index-url https://download.pytorch.org/whl/cpu; \
        ;; \
      *) \
        echo "Unsupported MODEL_RUNTIME=$MODEL_RUNTIME; expected cpu or gpu" >&2; \
        exit 2; \
        ;; \
    esac; \
    python -m pip install -r requirements-models.txt

COPY . .

RUN mkdir -p /app/.cache/huggingface /app/.cache/paddlex /app/data/outputs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
