# Deployment baseline

Wiki Hami production runs three application processes from the same image:

- `api`: FastAPI ingress, validation, job creation, status API
- `worker`: Celery consumer that runs OCR/layout/stamp-signature inference and publishes artifacts
- `ui`: optional Streamlit engineering console

The API and worker share the same model cache and `.env`. Redis is used for the Celery broker/result backend and durable job-state storage. MinIO remains an external/shared S3 dependency.

## Required production topology

```text
Backend
  -> POST /api/v1/extract/minio
  -> Wiki Hami API
       -> Redis queue
            -> Wiki Hami worker
                 -> MinIO artifacts
                 -> Backend callback
```

A deployment with only `api` and `ui` will accept jobs and return `202`, but jobs will remain `queued` forever because no Celery process is consuming `wiki_hami_extraction`.

## Required production configuration

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu

WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_BUCKET=media
WIKI_HAMI_MINIO_SECURE=false

WIKI_HAMI_CELERY_BROKER_URL=redis://redis:6379/0
WIKI_HAMI_CELERY_RESULT_BACKEND=redis://redis:6379/1
WIKI_HAMI_CELERY_QUEUE=wiki_hami_extraction
WIKI_HAMI_JOB_STORE_BACKEND=redis
WIKI_HAMI_JOB_STORE_REDIS_URL=redis://redis:6379/2
WIKI_HAMI_JOB_STORE_TTL_SECONDS=604800

WIKI_HAMI_BACKEND_API_KEY=<shared-backend-ai-secret>
WIKI_HAMI_CALLBACK_URL=http://backend:8000/api/documents/ai/callback/
WIKI_HAMI_CALLBACK_TOKEN=<callback-secret>
```

Use the actual Docker DNS name and internal port for Backend/Redis/MinIO in the deployment network.

## Worker

The worker command is:

```bash
python run.py --worker
```

It starts Celery with concurrency `1` and explicitly consumes the configured `WIKI_HAMI_CELERY_QUEUE` (default `wiki_hami_extraction`). Concurrency starts at one because all three inference backends are process-local and memory-heavy.

The worker uses the same image, model cache, output mount, environment and `wikio` external network as the API. The Compose worker healthcheck uses `celery inspect ping`; it must not inherit the API HTTP healthcheck because the worker does not listen on port 8000.

## Compose / GitLab

The root `compose.yaml` is the file used by the current GitLab deploy command:

```bash
docker compose up --build -d
```

It now defines `api`, `worker`, and `ui`. The CI pipeline therefore does not need a separate Celery launch command; recreating the Compose project starts the worker automatically.

After deployment verify:

```bash
docker compose ps
docker compose logs --tail=100 worker
docker compose exec worker celery -A app.jobs.celery_app:celery_app inspect ping
docker compose exec worker getent hosts redis
docker compose exec worker getent hosts minio
```

Expected worker startup includes the task `wiki_hami.process_minio_document`, queue `wiki_hami_extraction`, a successful Redis connection, and `celery@<hostname> ready.`

## Runtime sequence

1. Backend sends `POST /api/v1/extract/minio` with `X-API-Key`.
2. API writes job state `queued` to Redis and enqueues the Celery task.
3. Worker consumes the task and updates the job to `processing`.
4. Worker runs inference and writes `OCR.txt`, `layout.json`, and per-module artifacts to MinIO.
5. Worker records `completed` or `failed`.
6. Worker POSTs the terminal callback to Backend with `Authorization: Bearer <token>`.
7. Backend updates its document state.

See `docs/backend-async-contract.md` for the complete request/callback contract.
