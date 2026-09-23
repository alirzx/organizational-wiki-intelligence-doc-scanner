# Extraction V1 deployment

## Runtime topology

The normal deployment remains one FastAPI process plus optional Streamlit. MinIO is an external/shared S3 dependency; Wiki Hami does not require MinIO to run in the same Compose project, but the API container must be able to reach the configured S3 endpoint.

```text
Django / Celery -> Wiki Hami API <-> MinIO/S3
                           |
                    three CPU models
```

Keep one Uvicorn worker while all models are process-local. Every additional process loads another OCR, layout and RF-DETR model set.

## Local Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python run.py --api
```

Second terminal:

```bash
source .venv/bin/activate
python run.py --web
```

The base requirements include the official `minio` Python SDK.

## Real CPU model dependencies

Install in this order:

```bash
python -m pip install -r requirements-paddle-cpu.txt \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements-torch-cpu.txt \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-models.txt
```

Select:

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

## MinIO configuration

The current Backend contract sends image URLs such as:

```text
http://minio:9000/media/documents/123/images/page-001.jpg
```

Therefore server configuration should match that URL authority/bucket:

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
WIKI_HAMI_MINIO_BUCKET=media
```

For local development through the host-exposed S3 port:

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=192.168.4.209:9002
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://192.168.4.209:9002
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
WIKI_HAMI_MINIO_BUCKET=media
```

The Console port (for example 9003) is not used by Wiki Hami.

`MINIO_PUBLIC_BASE_URL` is also the strict validation policy for backend-provided image URLs. The API never treats those URLs as arbitrary HTTP download targets; bucket/object bytes are read through the configured MinIO SDK client.

Credentials belong in deployment secrets/GitLab `ENV_FILE`, never source control.

### Required MinIO permissions

The AI account is no longer read-only. Product `/extract/minio` writes artifacts and cleans old AI-owned prefixes before a retry. The credential therefore needs the equivalent of:

```text
ListBucket
GetObject
PutObject
DeleteObject
```

Scope delete permission to the Wiki Hami AI-owned output area where possible:

```text
documents/*/OCR/*
documents/*/Figure-Table/*
documents/*/Stamp-Signature/*
```

`documents/*/OCR.txt` also needs write permission. Backend-owned `images/`, `main.txt`, and `original.pdf` are never deleted by the application.

## Docker image

No model/device Dockerfile change is required. The MinIO SDK is already installed through the Python requirements.

```bash
docker build -t wiki-hami-extraction:0.3.0 .
```

The image exposes 8000/8501 and uses `/api/v1/health` for liveness. Liveness does not load models or contact MinIO. Use `/api/v1/storage/minio/health` when storage connectivity must be checked explicitly.

## Compose / GitLab deployment

The existing DevOps-owned Compose and GitLab CI files remain structurally unchanged. Compose loads `.env`; GitLab copies the `ENV_FILE` variable to the server before running `docker compose up --build -d`.

When MinIO endpoint/bucket/credentials change, update GitLab `ENV_FILE` and start a new pipeline on `main` so the new `.env` is copied and containers are recreated with the new configuration.

Do not modify Compose network ownership from application code. The API container simply requires DNS/network reachability to `minio:9000` on the DevOps-provided network.

## Production versus local storage browser

The Backend product integration needs:

```text
POST /api/v1/extract/minio
```

The isolated model and detailed inspection endpoints are engineering surfaces. Streamlit additionally uses MinIO health/list/object-proxy routes for browsing and preview.

Set:

```env
WIKI_HAMI_MINIO_BROWSER_ENABLED=false
```

when MinIO list/object proxy endpoints should not be exposed in production. If Streamlit is intentionally deployed as an internal engineering console, the flag may remain enabled behind appropriate network/access controls.

## First run

1. API starts without loading model weights.
2. `/api/v1/health` becomes healthy without MinIO/model warm-up.
3. `/api/v1/storage/minio/health` verifies bucket access.
4. The first product request reads page objects, validates/prepares images, then lazily initializes model backends as needed.
5. After all model modules succeed, the artifact publisher cleans only AI-owned module prefixes and writes JSON/text/crop artifacts.
6. Only after persistence completes does `/extract/minio` return product success.
7. Later requests reuse on-disk weights and process-local model instances.

## Operational notes

Per-image source objects are bounded by `max_upload_bytes` before full object read. Pillow enforces image decode/pixel validation after acquisition. Product writes are deterministic and retry-safe at the object-key level.

The API does not currently provide end-user authentication/rate limiting itself. Keep product and dev-storage endpoints behind the deployment network/API gateway appropriate to your environment.

For very large documents, the current request still retains prepared page/source images until orchestration and publishing finish. Monitor process RAM under realistic page counts; bounded page streaming/release is a future performance optimization rather than part of this contract refactor.
