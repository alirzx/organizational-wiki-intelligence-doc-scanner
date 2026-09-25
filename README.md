# Wiki Hami — Extraction V1

Production-oriented document extraction service for the Wiki Hami organizational-wiki pipeline. This repository implements **Step 1 only**: page acquisition, OCR paragraph extraction, figure/table detection, stamp/signature detection, deterministic artifact persistence, and a unified document layout for downstream reconstruction.

Template generation, document linking, semantic/RAG stages, table-cell extraction, signature identity, and final wiki generation are intentionally outside this repository.

## Production status

The production path is asynchronous:

```text
Backend / Django
      |
      | POST /api/v1/extract/minio + X-API-Key
      v
Wiki Hami FastAPI
      |
      | create job + enqueue
      v
Redis / Celery queue
      |
      v
Wiki Hami worker (concurrency=1)
      |
      +--> PaddleOCR                -> paragraph
      +--> PP-DocLayoutV3           -> figure, table
      +--> RF-DETR                  -> stamp, signature
      |
      v
ArtifactPublisher
      |
      +--> per-module page artifacts
      +--> OCR.txt
      +--> layout.json (unified layout v2)
      v
MinIO
      |
      | terminal callback + Bearer token
      v
Backend -> document READY / FAILED
```

The public production request schema is stable. `POST /api/v1/extract/minio` returns `202 Accepted` with a durable `job_id`; processing continues in the Celery worker. The Backend receives terminal state through the callback and may reconcile with `GET /api/v1/jobs/{job_id}`.

## Extraction outputs

Canonical object types:

- `paragraph` — PaddleOCR text detection/recognition + Wiki Hami paragraph grouping
- `table` — PP-DocLayoutV3
- `figure` — PP-DocLayoutV3
- `stamp` — RF-DETR
- `signature` — RF-DETR

All public geometry is restored to `exif_corrected_source_pixels`, meaning coordinates refer to the source page after EXIF display orientation and before shared model resize.

## MinIO contract

Default bucket: `media`.

```text
media/
└── documents/
    └── {document_id}/
        ├── original.pdf                    # Backend-owned
        ├── main.txt                        # Backend-owned
        ├── images/                         # Backend-owned AI inputs
        │   ├── page-001.jpg
        │   └── ...
        ├── OCR/                            # AI-owned
        │   ├── page-001.json.txt
        │   ├── page-001-text.txt
        │   └── ...
        ├── Figure-Table/                   # AI-owned
        │   ├── page-001.json.txt
        │   ├── page-001-table-001.png
        │   ├── page-001-figure-001.png
        │   └── ...
        ├── Stamp-Signature/                # AI-owned
        │   ├── page-001.json.txt
        │   ├── page-001-stamp-001.png
        │   ├── page-001-signature-001.png
        │   └── ...
        ├── OCR.txt                         # AI-owned document OCR aggregate
        └── layout.json                     # AI-owned unified document layout v2
```

`layout.json` is one file per document and contains every detected paragraph/table/figure/stamp/signature grouped by page, including source-page dimensions, `bbox`, optional `polygon`, confidence, metadata, provenance, artifact references, and a deterministic per-page geometric `reading_order`.

Detailed storage and layout contracts: [docs/minio.md](docs/minio.md).

## Production API

### Submit document

```http
POST /api/v1/extract/minio
X-API-Key: <shared-secret>
Content-Type: application/json
```

```json
{
  "document_id": "52",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/52/images/page-001.jpg",
      "page_id": "52:p1",
      "page_number": 1
    }
  ]
}
```

Immediate response:

```http
202 Accepted
```

```json
{
  "job_id": "<job-id>",
  "document_id": "52",
  "status": "queued"
}
```

### Recover job state

```http
GET /api/v1/jobs/{job_id}
X-API-Key: <shared-secret>
```

Terminal success includes `status: completed`, MinIO output paths, and callback delivery state.

### Backend callback

On success the worker sends:

```json
{
  "job_id": "<job-id>",
  "document_id": "52",
  "status": "completed",
  "result": {
    "ocr": "documents/52/OCR.txt",
    "layout": "documents/52/layout.json",
    "ocr_dir": "documents/52/OCR/",
    "figure_table_dir": "documents/52/Figure-Table/",
    "stamp_signature_dir": "documents/52/Stamp-Signature/"
  }
}
```

See [Backend async contract](docs/backend-async-contract.md) for the authoritative integration contract.

## Engineering APIs

These surfaces are for debugging, evaluation, and the Streamlit engineering console:

- `POST /api/v1/ocr`
- `POST /api/v1/figure-table`
- `POST /api/v1/stamp-signature`
- `POST /api/v1/extract` — multipart uploaded pages
- `POST /api/v1/extract/minio/inspect` — synchronous detailed MinIO inspection
- `GET /api/v1/storage/minio/health`
- `GET /api/v1/storage/minio/objects`
- `GET /api/v1/storage/minio/object`

Swagger/OpenAPI: `/docs`.

## Repository map

```text
app/api/             FastAPI routes, auth dependencies, request parsing
app/jobs/            Celery app, task, durable job store, callback delivery
app/storage/         MinIO URL policy and authenticated object operations
app/artifacts/       deterministic MinIO artifact publisher + layout v2
app/preprocessing/   decode, EXIF/RGB, resize, geometry transforms
app/modules/         OCR, figure/table, stamp/signature backends and adapters
app/orchestration/   per-page/all-module execution and aggregation
app/schemas/         canonical Pydantic contracts
app/core/            settings and process-local runtime registry
ui/                  Streamlit engineering inspector
tests/               unit/integration regression suite
deployment/          production Compose/operator notes
docs/                architecture and integration documentation
```

## Local setup

Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

For real CPU models:

```bash
python -m pip install -r requirements-paddle-cpu.txt \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements-torch-cpu.txt \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-models.txt
```

Local processes:

```bash
python run.py --api
python run.py --worker
python run.py --web
```

The production endpoint needs Redis/job-store configuration and a running worker. For synchronous engineering inspection only, the API/UI can still be used without the product queue path.

## Required production configuration

At minimum configure real model backends, MinIO, Redis, Backend API authentication, and callback authentication:

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr

WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_BUCKET=media
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>

WIKI_HAMI_CELERY_BROKER_URL=redis://redis:6379/0
WIKI_HAMI_CELERY_RESULT_BACKEND=redis://redis:6379/1
WIKI_HAMI_CELERY_QUEUE=wiki_hami_extraction
WIKI_HAMI_JOB_STORE_BACKEND=redis
WIKI_HAMI_JOB_STORE_REDIS_URL=redis://redis:6379/2

WIKI_HAMI_BACKEND_API_KEY=<shared-backend-ai-secret>
WIKI_HAMI_CALLBACK_URL=http://<backend-service>:8000/api/documents/ai/callback/
WIKI_HAMI_CALLBACK_TOKEN=<shared-callback-secret>
```

Never commit real credentials.

## Docker / deployment

The application image is shared by API, worker, and optional UI. The Compose project uses the external `wikio` network and a persistent model cache.

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 worker
```

The worker is required in production; without it, requests are accepted but remain queued. Current worker concurrency is intentionally `1` because model instances are process-local and memory-heavy.

Deployment runbook: [docs/deployment.md](docs/deployment.md).

## Verification

```bash
pytest -q
```

After deployment, verify all of the following:

1. API health passes.
2. Worker is connected to Redis and consumes `wiki_hami_extraction`.
3. MinIO health succeeds.
4. A real Front/Backend upload moves AI job `queued -> processing -> completed`.
5. MinIO contains per-page artifacts, `OCR.txt`, and `layout.json`.
6. `GET /jobs/{job_id}` reports `callback_delivered: true`.
7. Backend document transitions to `ready` and stores the callback `result` paths.

## Documentation

Start with [docs/README.md](docs/README.md).

- [Architecture](docs/architecture.md)
- [Production workflows](docs/workflows.md)
- [API reference](docs/api.md)
- [Backend async contract](docs/backend-async-contract.md)
- [MinIO + layout v2 contract](docs/minio.md)
- [Canonical contracts](docs/contracts.md)
- [Models and inference](docs/models.md)
- [Deployment runbook](docs/deployment.md)
