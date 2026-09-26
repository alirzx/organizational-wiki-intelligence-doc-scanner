# Production Deployment

Wiki Hami runs API and worker from the same image; Streamlit is optional.

```text
Backend
  -> FastAPI /extract/minio
       -> Redis queue
            -> Celery worker
                 -> three CPU extraction modules
                 -> MinIO artifacts
                 -> Backend callback
```

## Required services

- `api` — request validation, durable job creation, job-status API
- `worker` — Celery extraction consumer and callback sender
- `ui` — optional internal engineering console
- external/shared Redis
- external/shared MinIO
- external Docker network `wikio`

A running worker is mandatory for production. API-only deployment can accept jobs but cannot process them.

## Configuration

Use `.env.example` as the setting reference. Production requires real model backends, MinIO credentials, Redis URLs, Backend API key, callback URL, and callback token.

Core values:

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_ENABLE_MKLDNN=false
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr

WIKI_HAMI_MAX_PAGES_PER_DOCUMENT=200
WIKI_HAMI_MODULE_TIMEOUT_SECONDS=360

WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_BUCKET=media

WIKI_HAMI_CELERY_BROKER_URL=redis://redis:6379/0
WIKI_HAMI_CELERY_RESULT_BACKEND=redis://redis:6379/1
WIKI_HAMI_CELERY_QUEUE=wiki_hami_extraction
WIKI_HAMI_JOB_STORE_REDIS_URL=redis://redis:6379/2

WIKI_HAMI_BACKEND_API_KEY=<secret>
WIKI_HAMI_CALLBACK_URL=http://<backend-service>:8000/api/documents/ai/callback/
WIKI_HAMI_CALLBACK_TOKEN=<secret>
WIKI_HAMI_CALLBACK_TIMEOUT_SECONDS=15
WIKI_HAMI_CALLBACK_MAX_ATTEMPTS=3
```

Use the actual Backend DNS alias reachable from the worker.

Pinned CPU OCR runtime:

```text
PaddlePaddle 3.2.2
PaddleOCR    3.7.0
PaddleX      3.7.2
```

## Start / update

Current root Compose:

```bash
docker compose up -d --build
docker compose ps
```

Production-image Compose:

```bash
docker compose -f deployment/compose.prod.yaml up -d
```

The worker command is `python run.py --worker`; it consumes the configured extraction queue with concurrency `1`.

When requirements change, rebuild the image before recreating API/worker so the new dependency pins are installed.

## Verify

```bash
docker compose ps
docker compose logs --tail=100 worker
docker compose exec -T worker celery -A app.jobs.celery_app:celery_app inspect ping
docker compose exec -T worker getent hosts redis
docker compose exec -T worker getent hosts minio
```

Runtime verification after the OCR compatibility update:

```bash
docker compose exec -T worker python - <<'PY'
import paddle, paddleocr, paddlex
from app.core.config import get_settings
s = get_settings()
print("paddle:", paddle.__version__)
print("paddleocr:", paddleocr.__version__)
print("paddlex:", paddlex.__version__)
print("ocr_enable_mkldnn:", s.ocr_enable_mkldnn)
print("module_timeout:", s.module_timeout_seconds)
print("callback_timeout:", s.callback_timeout_seconds)
PY
```

Then submit one real Front/Backend document and confirm:

```text
queued -> processing -> completed
callback_delivered=true
Backend document status=ready
```

The successful callback uses top-level `result`; the AI job-status endpoint uses `outputs`.

For the full deployment/triage procedure see [`docs/deployment.md`](../docs/deployment.md). For request/callback details see [`docs/backend-async-contract.md`](../docs/backend-async-contract.md).
