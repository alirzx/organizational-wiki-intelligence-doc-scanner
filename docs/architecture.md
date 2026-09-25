# Extraction V1 Architecture

## Scope

Wiki Hami Extraction V1 converts raster document pages into five canonical object types: `paragraph`, `table`, `figure`, `stamp`, and `signature`. It also persists deterministic per-page artifacts, document OCR text, and a unified document layout for downstream reconstruction.

Out of scope: template generation, document linking, RAG/LLM mapping, table-cell extraction, signature identity, stamp interpretation, and final wiki generation.

## Production topology

```text
Frontend
   |
   v
Backend / Django + Celery
   |
   | stores original.pdf, main.txt, images/* in MinIO
   | POST /api/v1/extract/minio + X-API-Key
   v
Wiki Hami FastAPI
   |
   | validate request + MinIO URL policy
   | create durable job record
   | enqueue Celery task
   v
Redis
   |  broker /0
   |  result backend /1
   |  job-state store /2
   v
Wiki Hami Celery worker (concurrency=1)
   |
   +--> acquire MinIO page bytes
   +--> shared preprocessing
   +--> OCR / PaddleOCR
   +--> Figure-Table / PP-DocLayoutV3
   +--> Stamp-Signature / RF-DETR
   |
   v
ExtractionOrchestrator
   |
   v
ArtifactPublisher
   |
   +--> OCR/
   +--> Figure-Table/
   +--> Stamp-Signature/
   +--> OCR.txt
   +--> layout.json v2
   v
MinIO
   |
   | terminal callback + Bearer token
   v
Backend callback endpoint
   |
   v
READY / FAILED
```

The FastAPI request is intentionally short-lived. It returns `202 Accepted` after queueing; inference and persistence run only in the worker.

## Component boundaries

### Backend

Owns product orchestration and source artifacts:

- original upload/document state;
- page rendering/scanning;
- `original.pdf`, `main.txt`, `images/*`;
- AI job ID persistence;
- terminal callback handling and document state transition.

### FastAPI API

Owns ingress and job submission:

- validates `MinioDocumentRequest`;
- validates Backend API key;
- validates page URL bucket/path policy;
- creates `queued` job state;
- sends task to configured Celery queue;
- exposes recovery endpoint `GET /jobs/{job_id}`.

The API process does not perform production model inference for `/extract/minio`.

### Celery worker

Owns production processing:

- transitions job to `processing`;
- acquires pages from MinIO;
- runs all three modules;
- requires full-document success;
- publishes deterministic artifacts;
- transitions job to `completed` or `failed`;
- sends bounded-retry terminal callback.

Worker concurrency is currently `1` because inference models are process-local and memory-heavy.

### Redis

Default logical separation:

```text
redis://redis:6379/0   Celery broker
redis://redis:6379/1   Celery result backend
redis://redis:6379/2   durable AI job-state store
```

Job-store TTL defaults to seven days and is a recovery/reconciliation mechanism, not the system of record for Backend document state.

### MinIO

Shared object storage. The AI service reads Backend-owned page images and writes only AI-owned artifact keys. MinIO URLs received in requests are parsed and validated; model code never performs arbitrary URL downloads.

## Shared preprocessing

Both uploaded/debug and MinIO-backed pages converge on the same preparation path:

```text
bytes
 -> size/decode validation
 -> Pillow decode
 -> EXIF transpose
 -> RGB
 -> source image metadata
 -> bounded long-edge resize
 -> PreparedPage
```

Public geometry always refers to the EXIF-corrected source page before shared resize: `exif_corrected_source_pixels`.

Adapters restore model-space detections back to this coordinate system before creating `DetectedObject`.

## Model modules

```text
PreparedPage
  |
  +--> OCR service
  |      PaddleOCR detector + recognizer
  |      -> OCR lines
  |      -> geometric paragraph grouping
  |      -> paragraph DetectedObject
  |
  +--> Figure/Table service
  |      PP-DocLayoutV3
  |      -> figure/table DetectedObject
  |
  +--> Stamp/Signature service
         RF-DETR
         -> stamp/signature DetectedObject
```

Each module returns a `ModulePageResponse`. The orchestrator preserves the module-specific responses for artifact persistence and also builds the merged detailed response used by engineering endpoints.

## Artifact boundary

`ArtifactPublisher` is the only component that defines AI-owned deterministic product artifacts.

```text
documents/{document_id}/
├── OCR/
├── Figure-Table/
├── Stamp-Signature/
├── OCR.txt
└── layout.json
```

`layout.json` is currently `wiki-hami.layout.v2`. It merges all five object types per page and preserves source geometry, text where available, confidence, metadata, provenance, artifact references, and deterministic per-page reading order.

See [minio.md](minio.md) for the exact schema.

## Unified layout and reconstruction

`layout.json` is designed as the downstream reconstruction index:

```text
one document
  -> pages[]
       -> source width/height + coordinate space
       -> objects[]
            paragraph | table | figure | stamp | signature
            bbox
            optional polygon
            reading_order
            content/metadata/provenance
            artifact references
```

For visual reconstruction, consumers should position objects using `bbox`/`polygon` and page dimensions. `reading_order` is a deterministic geometric ordering, not a semantic layout model. Overlapping marks such as stamps/signatures must not be repositioned merely to satisfy reading order.

Current reading-order key is top-to-bottom, then left-to-right, with deterministic geometry/type/object-ID tie breakers.

## Retry and idempotency

Artifact keys are deterministic. Before a successful republish the AI service deletes only:

```text
documents/{id}/OCR/
documents/{id}/Figure-Table/
documents/{id}/Stamp-Signature/
```

Then it rewrites module artifacts. `OCR.txt` and `layout.json` are overwritten by object key. Backend-owned source objects are never deleted or modified.

This avoids stale visual crops after retries while preserving source data.

## Failure boundaries

Production publication occurs only when the orchestrator reports full success. A module failure prevents successful artifact publication for that run and produces a terminal failed job.

Detailed engineering endpoints may expose partial results, but product `/extract/minio` is an asynchronous all-required-modules contract.

Callback delivery failure does not rewrite a completed extraction as failed. The job remains completed and Backend can reconcile with `GET /jobs/{job_id}`.

## Security boundaries

- Backend -> AI: `X-API-Key`.
- AI -> Backend callback: Bearer token.
- Secrets are environment/deployment values only.
- Supplied MinIO URLs must match configured scheme/authority/bucket and expected document image prefix.
- Object keys reject unsafe path segments.
- Storage-browser endpoints are engineering surfaces and can be disabled.

## Process lifecycle

Model/service instances are process-local and lazily loaded. A second worker process means a second set of model objects and higher RAM usage. Persistent framework caches avoid repeated weight downloads but do not avoid model initialization per process.
