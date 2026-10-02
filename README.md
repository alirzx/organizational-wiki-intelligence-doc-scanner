# Wiki Hami — Extraction V1

Production-oriented document extraction service for the Wiki Hami organizational-wiki pipeline. This repository implements **Step 1 only**: page acquisition, OCR paragraph extraction, figure/table detection, stamp/signature detection, deterministic artifact persistence, and a unified document layout for downstream reconstruction.

Template generation, document linking, semantic/RAG stages, table-cell extraction, signature identity, and final wiki generation are intentionally outside this repository.

## ML-based Content Integrity Grouping

Document-scoped grouping is an optional additive stage that joins OCR blocks into paragraphs, lists, and heading/list sections, including conservative consecutive-page continuations. It preserves the five existing object types and source-coordinate page geometry. Enable it with `WIKI_HAMI_GROUPING_ENABLED=true` and select `WIKI_HAMI_GROUPING_BACKEND=clustering` for unsupervised grouping without a model package, or `lightgbm` for a validated trained package at `WIKI_HAMI_GROUPING_MODEL_PATH`. Grouping remains disabled by default.

The clustering backend uses [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) on a sparse graph of adjacent OCR blocks. Distances combine normalized spacing, alignment, and optional semantic similarity; structural gates protect column, heading, paragraph, and page boundaries. Tune neighborhood distance with `WIKI_HAMI_GROUPING_CLUSTER_EPS` (default `0.45`) and density with `WIKI_HAMI_GROUPING_CLUSTER_MIN_SAMPLES` (default `2`). Noise blocks remain separate. Responses report `grouping.mode=clustered` and a `dbscan.v1` algorithm ID. Confidence represents clustering affinity, not calibrated supervised probability. This backend uses scikit-learn from the grouping dependencies and needs neither LightGBM weights nor Ollama.

Semantic similarity is independently optional. Set `WIKI_HAMI_SEMANTIC_FEATURES_ENABLED=true` and configure `OLLAMA_BASE_URL` plus `OLLAMA_EMBEDDING_MODEL`; `fallback` mode continues without semantics or returns exact heuristic output when the model is unavailable, while `fail_fast` surfaces a sanitized error. Ollama is never required for heuristic deployments.

Before: page-local OCR may split one paragraph or concatenate unrelated list items. After: `content_groups` records stable ordered members and one page-local span per page, `content-groups.json` persists the document view, and learned `OCR.txt` renders every cross-page group once. Limits include OCR-dependent atomic blocks, conservative page-boundary candidates, and model quality tied to representative reviewed annotations.

Commands: build data with `python scripts/build_grouping_dataset.py`, train with `python scripts/train_grouping_model.py`, evaluate with `python scripts/evaluate_grouping_model.py`, export review cases with `python scripts/export_grouping_review.py`, and benchmark with `python scripts/benchmark_grouping.py`.

Grouping now consumes retained OCR lines through canonical normalization and deterministic source-coordinate block IDs. Heuristic paragraph objects remain available when grouping is disabled or falls back. The feature contract is `grouping.features.v2`; packages built with the previous divergent training extractor must be retrained. Threshold precedence is explicitly supplied runtime configuration, package metadata, then application defaults. Grouping runs outside the request event loop with two bounded workers by default; mutable clustering state belongs to one document.

Candidate budgets cap optional work. Eligible adjacent, boundary, local-column, heading/body, and list-continuation candidates are retained even when they exceed that optional cap. Responses add `grouping.candidates` coverage/count diagnostics and populated `grouping.embedding` provider/cache diagnostics. Embeddings are requested only for participating blocks. Retries use configurable backoff and a 60-second document deadline; successful batches remain cached if a later batch fails. A deadline stops waiting without spawning replacement requests while bounded provider work is still in flight.

Install development and grouping dependencies for the complete test suite: `python -m pip install -r requirements-dev.txt -r requirements-grouping.txt`. Run `python -m pytest`. Native training tests use deterministic synthetic fixtures, which do not establish production accuracy.

Run a stage benchmark with `python scripts/benchmark_grouping.py --blocks 1000 --repeats 3`. Add `--backend lightgbm --model-package models/grouping/current` for native inference, `--semantics mock --warm-cache` for controlled cache measurements, or `--semantics ollama` for real provider timings. `--trace-memory` measures Python allocations separately; `--repeats 20` enables p95 reporting. Held-out document evaluation uses `python scripts/evaluate_grouping_model.py input.jsonl output.json --documents`, optionally with `--model-package`. Each document supplies `document_id`, canonical block records, and `expected_groups` containing `member_ids` and `group_type`. This evaluation compares membership partitions, not arbitrary group IDs.

Docker images install the grouping dependencies by default (`INSTALL_GROUPING=true`). API and worker mount `./models/grouping` read-only at `/app/models/grouping`; production Compose uses `${WIKI_HAMI_DATA_ROOT}/models/grouping` instead. Model packages stay outside the image, so they can be deployed without downloading the OCR dependencies again.

To activate the supervised LightGBM backend, select `WIKI_HAMI_GROUPING_BACKEND=lightgbm` and provide reviewed training pairs or an already trained package. With a reviewed JSONL dataset available, train and validate the package before deploying it:

```bash
python -m pip install -r requirements-grouping.txt
python scripts/train_grouping_model.py --dataset data/grouping/pairs.jsonl --output-dir models/grouping/current
```

Set `WIKI_HAMI_GROUPING_ENABLED=true` and `WIKI_HAMI_GROUPING_MODEL_PATH=models/grouping/current` in `.env`, then recreate the services. Verify that the extraction response reports `grouping.mode` as `learned` or `learned_without_semantics` and includes `grouping.model_package_id`. An enabled flag alone does not activate a missing model: `heuristic_fallback` means LightGBM did not run. `WIKI_HAMI_SEMANTIC_FAILURE_POLICY=fail_fast` makes missing or incompatible grouping packages fail the extraction instead of falling back.

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

API, worker, and optional UI use the same Dockerfile and build configuration. Compose rebuilds their images on every `docker compose up`, reusing unchanged build layers. The Compose project uses the external `wikio` network and a persistent model cache.

If `files.pythonhosted.org` cannot resolve on your network, set `WIKI_HAMI_PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` in `.env` to use the [Tsinghua PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/). The default is `https://pypi.org/simple`. Paddle and CPU Torch wheels still use their official repositories; Torch dependencies use the selected PyPI index. Build-time dependency validation runs `pip check`.

```bash
docker compose up -d
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
