# Extraction V1 Workflows

## Production document workflow

```text
Frontend upload
   |
   v
Backend stores source document and rendered pages in MinIO
   |
   v
POST /api/v1/extract/minio
   |  X-API-Key
   v
FastAPI validates request + image object prefixes
   |
   +--> create job: queued
   +--> enqueue wiki_hami.process_minio_document
   |
   v
202 {job_id, document_id, status:"queued"}

Celery worker
   |
   +--> set job: processing
   +--> read MinIO page images
   +--> prepare pages
   +--> run OCR + Figure/Table + Stamp/Signature
   +--> require full success
   +--> publish AI-owned artifacts
   +--> set job: completed
   +--> POST success callback {result:{...}}
   v
Backend validates callback token + job/document IDs
   |
   v
Document READY
```

On processing failure the worker stores terminal `failed` state and sends the failure callback with structured `error {code,message}`.

## Shared page preparation

```text
MinIO page URL
   -> strict configured-host/bucket/path validation
   -> authenticated MinIO SDK read
   -> byte/pixel validation
   -> Pillow decode
   -> EXIF transpose
   -> RGB conversion
   -> source geometry definition
   -> bounded resize for inference
   -> PreparedPage
```

Uploaded engineering pages enter after acquisition and use the same decode/preprocessing path.

## Per-page extraction

For every prepared page, the orchestrator executes three module services with timeout/failure isolation:

```text
PreparedPage
   |
   +--> OCR --------------------> paragraph objects
   +--> Figure/Table -----------> figure + table objects
   +--> Stamp/Signature --------> stamp + signature objects
   |
   v
preserved ModulePageResponse per module
   +
merged PageExtractionResponse for debug/inspection
```

The artifact publisher consumes the preserved module responses so raw per-module artifacts do not have to be reconstructed from the flattened merged object list.

## Artifact publication

For each successful page:

```text
OCR
  -> OCR/page-NNN.json.txt
  -> OCR/page-NNN-text.txt

Figure/Table
  -> Figure-Table/page-NNN.json.txt
  -> zero or more table/figure PNG crops

Stamp/Signature
  -> Stamp-Signature/page-NNN.json.txt
  -> zero or more stamp/signature PNG crops
```

After all pages:

```text
OCR.txt      aggregate page OCR text
layout.json  unified document layout v2
```

`layout.json` combines all five canonical object types page by page and links visual objects back to their deterministic crop artifacts.

## Unified layout workflow

For each page, the publisher:

1. reads successful OCR, Figure/Table, and Stamp/Signature module results;
2. filters to the five public canonical object types;
3. preserves each object's full canonical fields (`bbox`, optional `polygon`, text, metadata, provenance, confidence);
4. attaches module name and artifact paths;
5. merges all objects into one page list;
6. sorts deterministically by `bbox.y1`, `bbox.x1`, then geometry/type/object-ID tie breakers;
7. assigns `reading_order` from 1..N;
8. records page source width/height, coordinate space, image metadata, transform metadata, and object counts.

This order is geometric. Preview reconstruction should use geometry as the source of truth, especially for overlapping stamps/signatures.

## Callback workflow

Success:

```text
worker completed
  -> POST WIKI_HAMI_CALLBACK_URL
  -> Authorization: Bearer <token>
  -> {job_id, document_id, status:"completed", result:{paths...}}
  -> Backend HTTP 2xx
  -> AI job callback_delivered=true
```

Failure:

```text
worker failed
  -> {job_id, document_id, status:"failed", error:{code,message}}
```

Callback delivery uses bounded retry/backoff. Delivery failure is recorded as callback metadata; it does not turn a successfully extracted job into a failed extraction.

## Recovery workflow

Backend can reconcile terminal state with:

```text
GET /api/v1/jobs/{job_id}
X-API-Key: <shared-secret>
```

Job-status response uses internal field `outputs` for MinIO path recovery. This is intentionally distinct from the Backend callback success field `result`.

## Retry / reprocessing workflow

A repeated run for the same document is deterministic:

- AI deletes only its three module prefixes;
- Backend-owned source objects remain untouched;
- module JSON/text/crops are regenerated;
- root `OCR.txt` and `layout.json` are overwritten.

This prevents stale crops when a rerun detects fewer visual objects.

## Engineering single-module workflow

```text
POST /api/v1/ocr
POST /api/v1/figure-table
POST /api/v1/stamp-signature
```

Each accepts one MinIO-backed page, runs one service, and returns `ModulePageResponse`. These routes do not define the product orchestration contract.

## Detailed local/upload workflow

`POST /api/v1/extract` accepts multipart uploaded images and returns `DocumentExtractionResponse`.

Optional:

```text
persist_outputs=true
```

When enabled, full success is required and the same `ArtifactPublisher` writes the production-style MinIO artifacts.

## Detailed MinIO inspection workflow

`POST /api/v1/extract/minio/inspect` accepts the production MinIO document body but runs synchronously for engineering inspection and returns detailed extraction data.

Optional query parameter:

```text
persist_outputs=true
```

This performs one inference pass and also writes the standard artifacts.

## Streamlit workflow

The Streamlit UI is an engineering console, not the product orchestrator. It supports local upload and MinIO inspection, overlays, canonical JSON inspection, and optional persistence.

## State model

```text
queued -> processing -> completed
                    \-> failed
```

`completed` means required model processing and MinIO publication succeeded. Callback delivery is tracked separately.
