# Extraction V1 architecture

## Scope

Extraction V1 turns raster document pages into canonical `paragraph`, `figure`, `table`, `stamp`, and `signature` detections. Template creation, document linking, RAG/LLM mapping, table-cell extraction, signature identity and wiki generation remain outside this repository.

## Product and local boundaries

```text
Backend / Celery                              Local inspector
      |                                             |
      | one document + all MinIO page URLs          +-- upload pages
      v                                             |
POST /api/v1/extract/minio                         +-- select MinIO objects
      |                                             |
      +----------------------+----------------------+
                             v
                   input acquisition adapter
                             |
                             v
                 shared validation/preprocessing
                             |
              +--------------+----------------+
              |              |                |
          PaddleOCR     PP-DocLayoutV3       RF-DETR
              |              |                |
              +------ canonical module results
                             |
                     ExtractionOrchestrator
                             |
                 +-----------+-----------+
                 |                       |
            merged result           ArtifactPublisher
            (debug/local)                 |
                                         v
                                  MinIO AI outputs
                                         |
                                         v
                                  small product status
```

The primary production endpoint is `/extract/minio`. The isolated `/ocr`, `/figure-table`, and `/stamp-signature` routes remain engineering/debug surfaces. `/extract` remains the multipart local workflow and `/extract/minio/inspect` is the detailed MinIO inspection workflow used by Streamlit.

## Acquisition boundary

MinIO remains isolated under `app/storage/`. Model backends do not know whether image bytes came from multipart upload or object storage. Both input paths converge on `prepare_page()`, which owns validation, Pillow decode, EXIF transpose, RGB conversion, bounded resize and construction of `PreparedPage`.

`WIKI_HAMI_MINIO_ENDPOINT` is the address used by Wiki Hami's MinIO SDK. `WIKI_HAMI_MINIO_PUBLIC_BASE_URL` is the host/port accepted in API-supplied URLs and used when canonical object URLs are built. On the server these are both currently expected to be `minio:9000` because Backend sends `http://minio:9000/media/...`. Local development can use the host-exposed S3 port instead.

## Orchestration boundary

`ExtractionOrchestrator` still runs all three module services with per-page concurrency and per-module timeout isolation. It now preserves each successful `ModulePageResponse` internally in addition to producing the existing merged `PageExtractionResponse` / `DocumentExtractionResponse`.

That separation is intentional:

- local/debug callers consume the merged detailed document response;
- `ArtifactPublisher` consumes the preserved module responses so each module can be serialized independently without reconstructing data from a flattened object list.

The model services themselves remain storage-agnostic.

## Artifact boundary

`app/artifacts/` owns deterministic product output generation. It receives a completed `DocumentRunResult`, creates raw JSON/text artifacts and visual crops, then writes them through `MinioStorageService`.

```text
documents/{document_id}/
├── OCR/
├── Figure-Table/
├── Stamp-Signature/
└── OCR.txt
```

Rules:

- OCR: per-page canonical JSON-as-TXT + per-page plain extracted text; no crops.
- Figure/Table: per-page canonical JSON-as-TXT + one PNG crop per detected figure/table.
- Stamp/Signature: per-page canonical JSON-as-TXT + one PNG crop per detected stamp/signature.
- document-level `OCR.txt`: aggregate compatibility text expected by Backend.
- crop geometry comes from canonical source-coordinate bounding boxes applied to `PreparedPage.source_image`.

The publisher never modifies Backend-owned `original.pdf`, `main.txt`, or `images/`.

## Retry/idempotency boundary

Celery may retry a document. Artifact names are therefore deterministic (`page-001-table-001.png`, etc.). Before publishing a successful rerun, Wiki Hami deletes only the three AI-owned module prefixes and then rewrites them. `OCR.txt` is overwritten by key.

This prevents stale crops when a later run produces fewer detections while avoiding bucket-wide or Backend-owned deletion.

## Service lifecycle

Settings, storage, artifact publisher and model services are process-local cached instances. Heavy model objects still initialize lazily on first prediction and remain CPU-configured. MinIO client construction is lazy. No model/device behavior changes as part of this refactor.

## Canonical contract

All public detection geometry remains `exif_corrected_source_pixels`. `ImageMetadata.source` records upload or MinIO acquisition provenance; MinIO provenance may include bucket, object key and ETag. Model `Provenance` remains separate and identifies module/backend/model/revision.

Detailed engineering responses retain `pages[]`, flattened `objects[]`, `object_counts` and processing status. The product endpoint intentionally returns only `document_id`, `status`, and optional `error` because extraction artifacts live in MinIO.

## Failure boundaries

Image acquisition/preparation happens before inference. Invalid MinIO URLs/images are rejected before model execution. Missing objects return 404, storage failures 502 and disabled/misconfigured storage 503.

The orchestrator still isolates module exceptions/timeouts. Detailed local/inspection flows can return `partial_success`. Product `/extract/minio` requires complete success before persistence; a failed module produces HTTP 500 and no product publishing attempt.

MinIO write/cleanup failures produce HTTP 502. Product success is returned only after all required artifact writes finish.

## Security

Backend URLs are not arbitrary HTTP fetch targets. The storage service validates scheme, configured public host/port, configured bucket and object path, then uses the authenticated MinIO client. Output object keys are also normalized and reject empty, `.` or `..` path segments.

Storage credentials remain environment-only. The product account needs read/list/write access and delete permission limited to the AI-owned artifact prefixes for retry cleanup.

The object listing/proxy endpoints remain internal Streamlit helpers controlled by `WIKI_HAMI_MINIO_BROWSER_ENABLED`.

## Parallelism and resources

Different model families can overlap and each real backend continues to serialize prediction on its shared model instance. Keep one Uvicorn worker unless memory measurements justify model duplication.

Current full-document preparation still retains prepared page images until document orchestration/publishing completes. A future bounded streaming/page-release optimization remains possible for very large documents, but this refactor does not alter model execution semantics or page concurrency.
