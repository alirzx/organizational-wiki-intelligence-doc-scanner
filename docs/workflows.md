# Extraction V1 workflows

## Shared page preparation

Both acquisition paths converge before model work:

```text
multipart UploadFile ----read bytes----+
                                      |
MinIO image_url -> validate -> SDK read+
                                      v
                              validate image bytes
                                      v
                         Pillow decode + EXIF transpose
                                      v
                      RGB + source geometry definition
                                      v
                     bounded long-edge resize (2500)
                                      v
                                PreparedPage
```

The same size/pixel/decode rules apply to uploaded and MinIO images. Public coordinates are source pixels after EXIF display orientation and before shared resize.

## Product document workflow

```text
Django / Celery
      |
      v
POST /api/v1/extract/minio
  document_id + pages[]
      |
      v
validate each MinIO URL
      |
      v
MinIO SDK read
      |
      v
prepare every page
      |
      v
ExtractionOrchestrator
  per page:
    OCR ------------------+
    Figure/Table ----------+--> preserved module results + merged debug result
    Stamp/Signature -------+
      |
      v
require full document success
      |
      v
ArtifactPublisher
      |
      +--> OCR/page-NNN.json.txt
      +--> OCR/page-NNN-text.txt
      +--> Figure-Table/page-NNN.json.txt + crops
      +--> Stamp-Signature/page-NNN.json.txt + crops
      +--> OCR.txt
      |
      v
HTTP 200 {document_id, status: success}
      |
      v
Celery continues
```

No callback or separate notification endpoint is used. Backend owns task orchestration and waits for this request.

## Engineering single-module workflow

```text
POST /ocr
POST /figure-table
POST /stamp-signature
```

Each accepts one validated MinIO image URL, runs only its selected model service and returns `ModulePageResponse`. These endpoints are retained for isolated debugging/evaluation, not as the main Backend integration.

## Local uploaded document

`POST /extract` retains multipart Streamlit/local evaluation. One or more uploaded pages are prepared and passed to the same `ExtractionOrchestrator`.

Default behavior returns the detailed `DocumentExtractionResponse` without product persistence.

Optional form field:

```text
persist_outputs=true
```

runs the same `ArtifactPublisher` after successful inference and still returns the detailed response.

## Local MinIO document inspection

`POST /extract/minio/inspect` accepts the same document/page request as production but returns the detailed canonical extraction response for Streamlit/testing.

Optional query parameter:

```text
persist_outputs=true
```

publishes the same production artifacts in the same inference pass.

## Streamlit workflow

The inspector keeps two input modes:

- Local upload → `/extract`
- MinIO browse/select/preview → `/extract/minio/inspect`

A **Persist product artifacts to MinIO** checkbox maps to the optional persistence flag on either inspection route. After either route returns, the detailed result workflow remains available: source/annotated comparison, per-module status, OCR/layout/mark views, canonical JSON, and JSON/ZIP export.

## Retry workflow

Celery may repeat `/extract/minio` for the same document. Before product persistence, only:

```text
documents/{document_id}/OCR/
documents/{document_id}/Figure-Table/
documents/{document_id}/Stamp-Signature/
```

are cleaned. Artifact names are deterministic, and document-level `OCR.txt` is overwritten. Backend-owned inputs remain untouched.

## Error/status workflow

Input acquisition/preparation failures reject the request before inference. The orchestrator still isolates module failures internally.

Detailed inspection routes may return `partial_success`. Product `/extract/minio` does not publish a partial run: any failed module produces HTTP 500 with `{status: failed}`. MinIO persistence failures produce HTTP 502. HTTP 200 means all modules and required output writes completed successfully.
