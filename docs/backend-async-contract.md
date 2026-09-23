# Backend ↔ AI Async Extraction Contract

## Purpose

The production MinIO document endpoint is asynchronous so Backend Celery does not hold an HTTP request open while OCR/layout/model inference runs. The request body is unchanged from the previous product contract.

## 1. Submit extraction

`POST /api/v1/extract/minio`

Production authentication header:

```http
X-API-Key: <shared-backend-ai-key>
Content-Type: application/json
```

Request body remains unchanged:

```json
{
  "document_id": "41",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/41/images/page-001.jpg",
      "page_id": "41:p1",
      "page_number": 1
    }
  ]
}
```

Each page URL must resolve to the configured bucket and must be under:

`documents/{document_id}/images/`

The API validates the request, creates a durable job record, enqueues Celery, and immediately returns:

```http
HTTP/1.1 202 Accepted
```

```json
{
  "job_id": "<unique-id>",
  "document_id": "41",
  "status": "queued"
}
```

Backend must not wait for OCR completion on this request.

## 2. Job states

AI manages these states:

- `queued`: accepted and waiting for worker execution
- `processing`: worker started processing
- `completed`: all required modules and MinIO writes completed
- `failed`: terminal processing/storage error

Recovery/polling endpoint:

`GET /api/v1/jobs/{job_id}`

Use the same `X-API-Key` header in production.

## 3. AI outputs in MinIO

AI writes only AI-owned outputs under `media/documents/{document_id}/`:

```text
media/documents/{document_id}/
├── main.txt                       # Backend-owned; AI never modifies it
├── images/                        # Backend-owned input pages
├── OCR.txt                        # AI: combined OCR text for all pages
├── layout.json                    # AI: document OCR layout / reading order
├── OCR/
│   ├── page-001.json.txt
│   ├── page-001-text.txt
│   └── ...
├── Figure-Table/
│   ├── page-001.json.txt
│   ├── page-001-table-001.png
│   ├── page-001-figure-001.png
│   └── ...
└── Stamp-Signature/
    ├── page-001.json.txt
    ├── page-001-stamp-001.png
    ├── page-001-signature-001.png
    └── ...
```

`OCR.txt` is created if absent and overwritten deterministically on a successful rerun.

`layout.json` is a document-level JSON artifact. It contains every OCR text block with pixel-space `bbox`, text/confidence, and a deterministic 1-based `reading_order` within each page.

Example:

```json
{
  "schema_version": "wiki-hami.layout.v1",
  "document_id": "41",
  "pages": [
    {
      "page_id": "41:p1",
      "page_number": 1,
      "blocks": [
        {
          "object_id": "...",
          "type": "paragraph",
          "bbox": {"x1": 120, "y1": 80, "x2": 900, "y2": 220},
          "reading_order": 1,
          "text": "...",
          "confidence": 0.97
        }
      ]
    }
  ]
}
```

Current reading order is deterministic top-to-bottom, then x-position within the same visual row. It can later be replaced by a more advanced document-layout reading-order model without changing the callback contract.

## 4. Completion callback

Callback URL is configured on the AI service through `WIKI_HAMI_CALLBACK_URL`; it is not added to the request body, so the existing request schema remains stable.

AI authenticates callback requests with:

```http
Authorization: Bearer <WIKI_HAMI_CALLBACK_TOKEN>
Content-Type: application/json
```

Success callback:

```json
{
  "job_id": "<unique-id>",
  "document_id": "41",
  "status": "completed",
  "outputs": {
    "ocr": "documents/41/OCR.txt",
    "layout": "documents/41/layout.json",
    "ocr_dir": "documents/41/OCR/",
    "figure_table_dir": "documents/41/Figure-Table/",
    "stamp_signature_dir": "documents/41/Stamp-Signature/"
  }
}
```

Failure callback:

```json
{
  "job_id": "<unique-id>",
  "document_id": "41",
  "status": "failed",
  "error": {
    "code": "RUNTIMEERROR",
    "message": "..."
  }
}
```

Callback delivery uses bounded retry/backoff. A callback delivery failure does not change an already-completed extraction back to `failed`; Backend can recover state with `GET /jobs/{job_id}`.

## 5. Required AI production configuration

```env
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_BUCKET=media

WIKI_HAMI_CELERY_BROKER_URL=redis://redis:6379/0
WIKI_HAMI_CELERY_RESULT_BACKEND=redis://redis:6379/1
WIKI_HAMI_JOB_STORE_BACKEND=redis
WIKI_HAMI_JOB_STORE_REDIS_URL=redis://redis:6379/2

WIKI_HAMI_BACKEND_API_KEY=<shared-secret>
WIKI_HAMI_CALLBACK_URL=http://<backend-service>/<callback-path>
WIKI_HAMI_CALLBACK_TOKEN=<shared-callback-secret>
```

The deployment must run both the FastAPI service and at least one Celery worker. For CPU-heavy model inference, worker concurrency should start at `1` unless memory measurements justify more workers.

## 6. Backend workflow

Recommended Backend Celery flow:

```text
Backend Celery
  -> POST /extract/minio
  <- 202 {job_id, status=queued}
  -> store job_id and continue without holding the HTTP request

AI worker
  -> processing
  -> model inference
  -> write OCR.txt / layout.json / per-page artifacts
  -> completed or failed
  -> POST callback to Backend

Backend
  -> update its document/job state from callback
  -> optionally GET /jobs/{job_id} for recovery/reconciliation
```
