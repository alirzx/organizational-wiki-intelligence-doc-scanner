# Wiki Hami — Extraction V1

Wiki Hami Extraction V1 converts document page images into canonical OCR paragraph, figure, table, stamp and signature detections. This repository is Step 1 only; template generation, document linking and later semantic/wiki stages are outside its scope.

## Current architecture

The product backend prepares page images in MinIO and sends one synchronous document request to Wiki Hami. The AI service validates and reads those pages through its authenticated MinIO client, runs all three CPU extraction modules, persists AI-owned artifacts back to MinIO, then returns a small success/failed status to the Backend Celery task.

```text
Django / Celery
      |
      v
POST /api/v1/extract/minio
      |
      v
MinIO page acquisition -> shared preprocessing
      |
      +--> OCR / PaddleOCR
      +--> Figure-Table / PP-DocLayoutV3
      +--> Stamp-Signature / RF-DETR
      |
      v
ArtifactPublisher
      |
      v
MinIO AI outputs
      |
      v
small status response
```

The three single-module APIs remain available for engineering/debug use. Local development supports both uploaded images and MinIO objects and can optionally persist the same production artifacts while still returning detailed inspection data.

See [architecture](docs/architecture.md), [workflows](docs/workflows.md), [API reference](docs/api.md), and [MinIO integration](docs/minio.md).

## APIs

Primary product/backend:

- `POST /api/v1/extract/minio` — one document + all MinIO page URLs; persists outputs and returns status

Engineering/debug:

- `POST /api/v1/ocr` — one MinIO page, detailed OCR response
- `POST /api/v1/figure-table` — one MinIO page, detailed layout response
- `POST /api/v1/stamp-signature` — one MinIO page, detailed mark response
- `POST /api/v1/extract` — multipart uploaded pages, detailed full response
- `POST /api/v1/extract/minio/inspect` — MinIO pages, detailed full response

Storage inspection for the local UI:

- `GET /api/v1/storage/minio/health`
- `GET /api/v1/storage/minio/objects`
- `GET /api/v1/storage/minio/object`

Swagger/OpenAPI is at `/docs`.

## Product MinIO layout

Input prepared by Backend:

```text
media/documents/{document_id}/images/page-001.jpg
media/documents/{document_id}/images/page-002.jpg
```

AI-owned outputs:

```text
documents/{document_id}/
├── OCR/
│   ├── page-001.json.txt
│   └── page-001-text.txt
├── Figure-Table/
│   ├── page-001.json.txt
│   ├── page-001-table-001.png
│   └── page-001-figure-001.png
├── Stamp-Signature/
│   ├── page-001.json.txt
│   ├── page-001-stamp-001.png
│   └── page-001-signature-001.png
└── OCR.txt
```

OCR does not produce crop images. Figure/Table and Stamp/Signature crops are generated from EXIF-corrected source images using canonical source-coordinate bounding boxes. Deterministic names plus AI-prefix cleanup make Celery retries idempotent without touching Backend-owned `original.pdf`, `main.txt`, or `images/`.

## Baseline models

| Module | Model | Device | Canonical output |
|---|---|---|---|
| OCR | `PP-OCRv5_server_det` + `arabic_PP-OCRv5_mobile_rec` | CPU | `paragraph` |
| Figure/Table | `PaddlePaddle/PP-DocLayoutV3` | CPU | `figure`, `table` |
| Stamp/Signature | `bluecopa/rf-detr-stamp-signature-detector` | CPU | `stamp`, `signature` |

The product refactor does not change model/device inference configuration.

## Repository structure

```text
run.py               unified local launcher
app/api/             FastAPI routes and request contracts
app/storage/         MinIO URL validation plus authenticated read/write operations
app/artifacts/       deterministic JSON/text/crop artifact generation and persistence
app/core/            environment settings and process-local service registry
app/preprocessing/   validation, decode, EXIF/RGB/resize and geometry transforms
app/modules/         OCR, layout and RF-DETR backends/services/adapters
app/orchestration/   multi-page/all-model execution, preserved module results and aggregation
app/schemas/         public Pydantic contracts
ui/                  Streamlit engineering inspector and exports
tests/               unit/integration contracts
deployment/          production-style Compose baseline
docs/                architecture, API, storage and deployment documentation
```

## Python 3.11 setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

For real CPU models install in this order:

```bash
python -m pip install -r requirements-paddle-cpu.txt \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements-torch-cpu.txt \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-models.txt
```

## Environment

Real model selection:

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

Server/product MinIO when Backend sends `http://minio:9000/media/...`:

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
WIKI_HAMI_MINIO_BUCKET=media
```

Local host-exposed MinIO can instead use `192.168.4.209:9002` for both endpoint and public base URL while keeping bucket `media`. Never commit real credentials. The AI account now needs read/list/write permission and delete permission for retry cleanup of AI-owned prefixes.

## Run locally

```bash
source .venv/bin/activate
python run.py --api
```

Second terminal:

```bash
source .venv/bin/activate
python run.py --web
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

The Streamlit inspector provides Local Upload and MinIO input modes, MinIO health/object browsing and preview, detailed canonical results, overlays and downloads. Enable **Persist product artifacts to MinIO** to run the same publisher used by `/extract/minio` without losing the detailed local response.

## Model caching

First real inference downloads missing model weights. Framework caches are persisted by Compose and later requests reuse in-memory model instances. MinIO objects are read from object storage per request; they are not copied into model-cache directories.

## Docker / deployment

No model image/device changes are required. Existing Compose/CI ownership remains unchanged; deployment continues to load `.env` and run the same API/UI containers.

```bash
docker compose up -d --build
```

Production should keep one Uvicorn worker while models are process-local. Disable `WIKI_HAMI_MINIO_BROWSER_ENABLED` when the Streamlit storage browser/proxy is not required.

## Tests

```bash
pytest -q
```

The suite uses mock models/network-free fixtures and covers URL policy, local/minio extraction, orchestration failure isolation, artifact layout, OCR crop exclusion, deterministic visual crops, provenance, transforms, adapters and UI helpers.

## Documentation

- [API reference](docs/api.md)
- [MinIO integration](docs/minio.md)
- [Architecture](docs/architecture.md)
- [Workflows](docs/workflows.md)
- [Models](docs/models.md)
- [Deployment](docs/deployment.md)
- [Contract notes](docs/contracts.md)
