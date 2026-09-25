# Backend ↔ AI Async Extraction Contract

This document is the authoritative production integration contract between the Backend and Wiki Hami Extraction V1.

## 1. Submit extraction

```http
POST /api/v1/extract/minio
X-API-Key: <shared-backend-ai-key>
Content-Type: application/json
```

Request body:

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

Rules:

- `document_id` must be a safe single path segment;
- page numbers and page IDs must be unique within the document request;
- each page URL must match configured MinIO host/port and bucket;
- each object key must be under `documents/{document_id}/images/`;
- maximum page count is controlled by `WIKI_HAMI_MAX_PAGES_PER_DOCUMENT`.

Immediate response:

```http
202 Accepted
```

```json
{
  "job_id": "<unique-id>",
  "document_id": "52",
  "status": "queued"
}
```

Backend must persist the returned `job_id` and must not hold the request open waiting for model completion.

## 2. AI job states

```text
queued -> processing -> completed
                    \-> failed
```

Meanings:

- `queued`: request validated, job stored, task submitted to Celery;
- `processing`: worker started the task;
- `completed`: all required modules succeeded and required MinIO artifacts were written;
- `failed`: terminal model/acquisition/persistence error.

## 3. Recovery endpoint

```http
GET /api/v1/jobs/{job_id}
X-API-Key: <shared-backend-ai-key>
```

Example completed response:

```json
{
  "job_id": "<job-id>",
  "document_id": "52",
  "status": "completed",
  "created_at": "...",
  "updated_at": "...",
  "outputs": {
    "ocr": "documents/52/OCR.txt",
    "layout": "documents/52/layout.json",
    "ocr_dir": "documents/52/OCR/",
    "figure_table_dir": "documents/52/Figure-Table/",
    "stamp_signature_dir": "documents/52/Stamp-Signature/"
  },
  "callback_delivered": true
}
```

Important: the job-status API uses `outputs`. The Backend success callback uses `result`.

## 4. MinIO ownership

Bucket default: `media`.

Backend-owned and never modified/deleted by AI:

```text
documents/{id}/original.pdf
documents/{id}/main.txt
documents/{id}/images/*
```

AI-owned:

```text
documents/{id}/OCR/*
documents/{id}/Figure-Table/*
documents/{id}/Stamp-Signature/*
documents/{id}/OCR.txt
documents/{id}/layout.json
```

The callback/job path values are **object keys inside bucket `media`**. Therefore the value is `documents/52/OCR.txt`, not `media/documents/52/OCR.txt`.

## 5. Unified layout artifact

`documents/{id}/layout.json` is document-level schema:

```text
wiki-hami.layout.v2
```

It contains all five object types:

```text
paragraph, table, figure, stamp, signature
```

Each page includes source dimensions/coordinate space, image and transform metadata, object counts, and a merged `objects[]` list. Each object preserves canonical geometry/content/metadata/provenance and adds per-page `reading_order` plus artifact references.

See [minio.md](minio.md) for the complete schema.

## 6. Success callback

Configured by `WIKI_HAMI_CALLBACK_URL`.

Authentication:

```http
Authorization: Bearer <WIKI_HAMI_CALLBACK_TOKEN>
Content-Type: application/json
```

Payload:

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

The key is `result`. Do not change it to `outputs` unless both Backend and AI contracts are versioned together.

Expected Backend behavior:

- validate Bearer token;
- locate document by `document_id`;
- require matching `job_id`;
- store `result` as document processing result;
- transition processing document to READY;
- return HTTP 2xx;
- treat duplicate callbacks to an already-terminal document idempotently.

## 7. Failure callback

```json
{
  "job_id": "<job-id>",
  "document_id": "52",
  "status": "failed",
  "error": {
    "code": "RUNTIMEERROR",
    "message": "..."
  }
}
```

`error` must contain both `code` and `message`.

## 8. Callback delivery semantics

The AI worker retries callback delivery with bounded retry/backoff according to:

```env
WIKI_HAMI_CALLBACK_TIMEOUT_SECONDS=10
WIKI_HAMI_CALLBACK_MAX_ATTEMPTS=3
```

Callback failure is tracked independently:

```text
callback_delivered=false
callback_error=<last delivery exception>
```

A completed extraction remains `completed` even if callback delivery fails. Backend can reconcile from the job-status endpoint.

## 9. Production configuration

```env
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
WIKI_HAMI_JOB_STORE_TTL_SECONDS=604800

WIKI_HAMI_BACKEND_API_KEY=<shared-secret>
WIKI_HAMI_CALLBACK_URL=http://<backend-service>:8000/api/documents/ai/callback/
WIKI_HAMI_CALLBACK_TOKEN=<shared-secret>
```

The production deployment requires both FastAPI and a Celery worker. Without a worker, submission can return `202` while jobs remain queued.

## 10. End-to-end acceptance criteria

For one real Front upload, integration is considered healthy when all are true:

1. Backend receives `202` and stores the AI `job_id`.
2. AI job moves `queued -> processing -> completed`.
3. Per-page artifacts, `OCR.txt`, and `layout.json` exist in MinIO.
4. `GET /jobs/{job_id}` returns `callback_delivered: true`.
5. Backend stores the callback `result` paths.
6. Backend document state becomes READY.
7. Backend error fields are empty.
