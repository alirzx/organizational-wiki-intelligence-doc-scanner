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

COPY requirements.txt requirements-models.txt requirements-paddle-cpu.txt requirements-torch-cpu.txt ./

ARG INSTALL_MODELS=true
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN if [ "$INSTALL_MODELS" = "true" ]; then \
      python -m pip install -r requirements-paddle-cpu.txt \
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/; \
    fi

RUN if [ "$INSTALL_MODELS" = "true" ]; then \
      python -m pip install --no-deps -r requirements-torch-cpu.txt \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Resolve CPU Torch dependencies with the application from the selected PyPI index.
ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN if [ "$INSTALL_MODELS" = "true" ]; then \
      python -m pip install --index-url "$PYPI_INDEX_URL" \
        -r requirements-torch-cpu.txt -r requirements-models.txt \
        -c requirements-paddle-cpu.txt; \
    else \
      python -m pip install --index-url "$PYPI_INDEX_URL" -r requirements.txt; \
    fi && python -m pip check

COPY requirements-grouping.txt ./
ARG INSTALL_GROUPING=true
RUN if [ "$INSTALL_GROUPING" = "true" ]; then \
      python -m pip install --index-url "$PYPI_INDEX_URL" -r requirements-grouping.txt; \
    fi && python -m pip check

COPY . .

RUN mkdir -p /app/.cache/huggingface /app/.cache/paddlex /app/data/outputs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
