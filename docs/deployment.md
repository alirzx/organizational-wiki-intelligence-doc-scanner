# Extraction V1 Deployment Runbook

## Production topology

Production requires three shared dependencies/process roles:

```text
Backend
   |
   v
Wiki Hami API ----> Redis ----> Wiki Hami worker
   |                              |
   |                              +--> CPU models
   |                              +--> MinIO writes
   |                              +--> Backend callback
   |
   +--> job-status API

MinIO <---------------------------> API/worker
```

The optional Streamlit UI is an engineering console and is not required for Backend product processing.

## Required application processes

- `api`: FastAPI ingress, validation, job creation, recovery API
- `worker`: Celery consumer, model inference, artifact publication, callback delivery
- `ui`: optional Streamlit engineering inspector

A deployment without `worker` may still accept `/extract/minio` and return `202`, but jobs will remain queued.

## Runtime configuration

Start from `.env.example` and keep production secrets out of Git.

### Models

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_OCR_ENABLE_MKLDNN=false
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

The pinned CPU OCR runtime baseline is PaddlePaddle 3.2.2 + PaddleOCR 3.7.0 + PaddleX 3.7.2. oneDNN/MKLDNN is disabled by default for OCR stability unless the exact target runtime has been regression-tested.

### Workload limits

```env
WIKI_HAMI_MAX_UPLOAD_BYTES=26214400
WIKI_HAMI_MAX_IMAGE_PIXELS=50000000
WIKI_HAMI_MAX_PAGES_PER_DOCUMENT=200
WIKI_HAMI_PREPROCESS_MAX_LONG_EDGE=2500
WIKI_HAMI_PAGE_CONCURRENCY=4
WIKI_HAMI_MODULE_TIMEOUT_SECONDS=360
```

`WIKI_HAMI_MODULE_TIMEOUT_SECONDS` is per module/per page; it is not a whole-document Celery time limit.

### MinIO

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
WIKI_HAMI_MINIO_BUCKET=media
```

The AI credential requires read/list/write plus delete permission for AI-owned module prefixes. Scope delete permission narrowly where possible.

### Redis / jobs

```env
WIKI_HAMI_CELERY_BROKER_URL=redis://redis:6379/0
WIKI_HAMI_CELERY_RESULT_BACKEND=redis://redis:6379/1
WIKI_HAMI_CELERY_QUEUE=wiki_hami_extraction
WIKI_HAMI_JOB_STORE_BACKEND=redis
WIKI_HAMI_JOB_STORE_REDIS_URL=redis://redis:6379/2
WIKI_HAMI_JOB_STORE_TTL_SECONDS=604800
```

### Backend integration

```env
WIKI_HAMI_BACKEND_API_KEY=<shared-backend-ai-secret>
WIKI_HAMI_CALLBACK_URL=http://<backend-service>:8000/api/documents/ai/callback/
WIKI_HAMI_CALLBACK_TOKEN=<shared-callback-secret>
WIKI_HAMI_CALLBACK_TIMEOUT_SECONDS=15
WIKI_HAMI_CALLBACK_MAX_ATTEMPTS=3
```

Use the actual Docker DNS alias visible from the AI worker. Do not assume a Compose container name and DNS alias are identical.

## Docker Compose

Root `compose.yaml` defines API, worker, and UI on external network `wikio` with shared persistent model cache.

```bash
docker compose up -d --build
docker compose ps
```

Production-style Compose is also provided under `deployment/compose.prod.yaml` for externally built images/data roots.

The worker command is:

```bash
python run.py --worker
```

It consumes configured queue `wiki_hami_extraction` with concurrency `1`.

## Model cache

Persist model caches across container recreation. Current Compose config provides persistent Hugging Face/Paddle/PaddleX cache locations. A cache hit avoids weight redownload but each worker process still initializes model objects in RAM.

Do not increase Uvicorn/Celery process counts casually: every additional inference process can duplicate model memory.

## External network

Compose expects:

```text
wikio
```

as an externally managed Docker network. Backend, Redis, and MinIO names used in `.env` must resolve on this network.

Useful checks:

```bash
docker compose exec -T worker getent hosts redis
docker compose exec -T worker getent hosts minio
docker compose exec -T worker getent hosts <backend-service>
```

## Health checks

API liveness:

```bash
curl -fsS http://localhost:8000/api/v1/health
```

MinIO connectivity:

```bash
curl -fsS http://localhost:8000/api/v1/storage/minio/health
```

Worker:

```bash
docker compose exec -T worker \
  celery -A app.jobs.celery_app:celery_app inspect ping
```

Worker logs should show configured queue, Redis connection, task registration, and `ready`.

## Deployment sequence

Recommended rollout:

```bash
git pull origin main
docker compose build
docker compose up -d --force-recreate
docker compose ps
```

When dependency pins change, force a fresh image rebuild so the image does not reuse an older dependency layer. If only env/secrets changed, containers still need recreation so processes receive the new values.

## Production acceptance test

Use a real Front upload, then verify:

1. Backend stores rendered pages in MinIO.
2. Backend receives `202 {job_id,status:"queued"}`.
3. Worker log shows `wiki_hami.process_minio_document[JOB_ID] received`.
4. Worker finishes successfully.
5. AI job endpoint reports `status: completed`.
6. MinIO contains module outputs, `OCR.txt`, and unified `layout.json`.
7. AI job reports `callback_delivered: true`.
8. Backend document is `ready`.
9. Backend `processing_result` contains the five returned output paths.
10. Backend error fields are empty.

## Useful job check

```bash
JOB=<job-id>
docker compose exec -T api python - "$JOB" <<'PY'
import os, sys, json, requests
r = requests.get(
    f"http://localhost:8000/api/v1/jobs/{sys.argv[1]}",
    headers={"X-API-Key": os.environ["WIKI_HAMI_BACKEND_API_KEY"]},
    timeout=10,
)
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
PY
```

Expected terminal success includes:

```json
{
  "status": "completed",
  "callback_delivered": true
}
```

## Logs

```bash
docker compose logs --tail=100 api
docker compose logs --tail=100 worker
docker compose logs -f worker
```

A Celery warning about running as root is currently an operational hardening item rather than an extraction-contract failure. Production hardening should eventually run the worker as a non-root container user.

## GitLab deployment

The repository GitLab pipeline copies the configured env file to the server and runs the Compose deployment on `main`. Treat the GitLab env file/CI variables as the production secret source of truth.

The current CI command rebuilds through root `compose.yaml`, so Dockerfile requirement changes are picked up during deployment. The CI file itself is intentionally unchanged by this runtime-compatibility work.

After a documentation-only commit, no runtime deployment is required unless the team wants repository/server revisions aligned. After application, dependency, or environment changes, rebuild/recreate and run the acceptance test above.

## Failure triage

### Job remains queued

Check worker existence, queue name, Redis DNS/connectivity, and worker logs.

### Extraction completed but Front stays processing

Check:

```text
callback_delivered
callback_error
callback URL DNS/path
callback token
Backend callback serializer contract
job_id/document_id match
```

### 401 callback

Callback authentication mismatch.

### 400 callback

Inspect Backend response body and serializer expectations. Current successful contract requires top-level `result`.

### Model/native failure

Treat separately from callback/network failures. Review worker traceback and identify module/page. Do not classify every native Paddle failure as concurrency unless reproduced with evidence.

For the current OCR stability baseline, verify all of the following before further diagnosis:

```text
PaddlePaddle 3.2.2
PaddleOCR 3.7.0
PaddleX 3.7.2
WIKI_HAMI_OCR_ENABLE_MKLDNN=false
```

### Security / operational notes

- Do not commit `.env` or secrets.
- Keep storage-browser routes internal or disable them in production.
- Restrict MinIO delete permission to AI-owned prefixes.
- Keep Backend/Redis/MinIO on trusted internal network paths.
- Consider non-root worker/container execution as the next deployment-hardening step.
