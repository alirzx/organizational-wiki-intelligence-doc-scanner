# Extraction V1 API Reference

Base prefix: `/api/v1`. Swagger/OpenAPI: `/docs`.

## Production integration

### Submit asynchronous MinIO document

```http
POST /api/v1/extract/minio
X-API-Key: <shared-backend-ai-key>
Content-Type: application/json
```

Request:

```json
{
  "document_id": "52",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/52/images/page-001.jpg",
      "page_number": 1,
      "page_id": "52:p1"
    },
    {
      "image_url": "http://minio:9000/media/documents/52/images/page-002.jpg",
      "page_number": 2,
      "page_id": "52:p2"
    }
  ]
}
```

Important fields:

| Field | Required | Meaning |
|---|---:|---|
| `document_id` | yes | Stable document identity and MinIO path segment |
| `document_metadata` | no | Arbitrary document metadata |
| `pages` | yes | One or more MinIO-backed pages |
| `pages[].image_url` | yes | URL matching configured MinIO authority/bucket/path policy |
| `pages[].page_number` | no | Integer >= 1; request order is fallback |
| `pages[].page_id` | no | Stable page ID; default derived from document/page number |
| `pages[].filename` | no | Optional filename override |
| `pages[].metadata` | no | Arbitrary page metadata |

Request validation rejects duplicate page numbers/IDs and pages outside `documents/{document_id}/images/`.

Response:

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

`/extract/minio` does not wait for inference. The Celery worker owns production processing and callback delivery.

### Job status / recovery

```http
GET /api/v1/jobs/{job_id}
X-API-Key: <shared-backend-ai-key>
```

States:

```text
queued | processing | completed | failed
```

Completed jobs may include:

```json
{
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

The job-status field is named `outputs`. The separate Backend callback success payload uses `result`.

## Production callback

The AI service sends the callback to configured `WIKI_HAMI_CALLBACK_URL`.

```http
Authorization: Bearer <WIKI_HAMI_CALLBACK_TOKEN>
Content-Type: application/json
```

Success:

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

Failure:

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

See [backend-async-contract.md](backend-async-contract.md).

## Product artifacts

Successful production jobs publish:

```text
documents/{document_id}/
├── OCR/
├── Figure-Table/
├── Stamp-Signature/
├── OCR.txt
└── layout.json
```

`layout.json` is `wiki-hami.layout.v2` and merges all five canonical object types per page. See [minio.md](minio.md).

## Engineering module endpoints

These are synchronous inspection/evaluation surfaces and are not the Backend product orchestration path.

### OCR

```http
POST /api/v1/ocr
```

Runs OCR text detection/recognition and paragraph grouping for one MinIO page. Returns `ModulePageResponse` with `paragraph` objects.

### Figure/Table

```http
POST /api/v1/figure-table
```

Runs PP-DocLayoutV3 and returns canonical `figure`/`table` objects.

### Stamp/Signature

```http
POST /api/v1/stamp-signature
```

Runs RF-DETR and returns canonical `stamp`/`signature` objects.

Typical request:

```json
{
  "document_id": "52",
  "image_url": "http://minio:9000/media/documents/52/images/page-001.jpg",
  "page_number": 1,
  "page_id": "52:p1",
  "page_metadata": {}
}
```

## Detailed multipart extraction

```http
POST /api/v1/extract
```

Multipart fields:

- repeated `images`;
- `document_id`;
- optional `document_metadata_json`;
- optional `pages_metadata_json`;
- optional `persist_outputs` boolean.

Returns `DocumentExtractionResponse` synchronously.

If `persist_outputs=true`, the same production artifact publisher runs after full success.

## Detailed MinIO inspection

```http
POST /api/v1/extract/minio/inspect
```

Accepts the same JSON body as production `/extract/minio`, but runs synchronously and returns detailed `DocumentExtractionResponse` for Streamlit/testing.

Optional:

```text
?persist_outputs=true
```

## Health / storage endpoints

```text
GET /api/v1/health
GET /api/v1/storage/minio/health
GET /api/v1/storage/minio/objects
GET /api/v1/storage/minio/object
```

`/health` is process/config liveness and does not warm models or verify MinIO.

Storage browser/list/object-proxy routes are engineering surfaces controlled by `WIKI_HAMI_MINIO_BROWSER_ENABLED`.

## Canonical object

Every `DetectedObject` includes:

```text
object_id
document_id
page_id
page_number
type
bbox
polygon (optional)
confidence
text (optional)
raw_text (optional)
metadata
provenance
```

Canonical types:

```text
paragraph | table | figure | stamp | signature
```

Coordinates use `exif_corrected_source_pixels`.

## Error semantics

### Submission endpoint

Common errors include:

- `401/403` style authorization failure from API-key dependency;
- `422` invalid request/document/page/URL policy;
- `503` queue/job-store/configuration failure.

Model processing errors occur asynchronously and are reflected in job state/callback, not as the original submission HTTP response.

### Engineering synchronous endpoints

Detailed synchronous routes may directly return validation, model, or MinIO errors because they execute work inside the request.
