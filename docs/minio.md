# MinIO integration

## Product contract

The primary Backend → AI integration is:

```text
POST /api/v1/extract/minio
```

The backend/Celery task prepares page images in MinIO and sends one document request containing all page URLs. Wiki Hami reads those images through its authenticated MinIO client, runs OCR + Figure/Table + Stamp/Signature extraction, writes AI-owned artifacts back to MinIO, and returns a small synchronous status response.

```text
Django / Celery
      |
      v
POST /api/v1/extract/minio
      |
      v
validated MinIO page acquisition
      |
      v
shared preprocessing
      |
      +--> OCR
      +--> Figure/Table
      +--> Stamp/Signature
      |
      v
artifact publisher
      |
      v
MinIO outputs
      |
      v
{"document_id":"123","status":"success"}
```

No callback is required at this stage because the Backend Celery task waits for the same request.

## Backend request

```json
{
  "document_id": "123",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
      "page_id": "123:p1",
      "page_number": 1
    },
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-002.jpg",
      "page_id": "123:p2",
      "page_number": 2
    }
  ]
}
```

Production bucket/path contract:

```text
bucket: media
input:  documents/{document_id}/images/
```

## AI-owned output structure

Wiki Hami owns only the following artifact locations:

```text
documents/{document_id}/
├── OCR/
│   ├── page-001.json.txt
│   ├── page-001-text.txt
│   ├── page-002.json.txt
│   └── page-002-text.txt
├── Figure-Table/
│   ├── page-001.json.txt
│   ├── page-001-table-001.png
│   ├── page-001-figure-001.png
│   └── ...
├── Stamp-Signature/
│   ├── page-001.json.txt
│   ├── page-001-stamp-001.png
│   ├── page-001-signature-001.png
│   └── ...
└── OCR.txt
```

Rules:

- OCR never stores cropped images.
- Every OCR page stores the canonical module JSON as UTF-8 JSON in `page-NNN.json.txt`.
- Every OCR page also stores plain extracted text in `page-NNN-text.txt`.
- `documents/{document_id}/OCR.txt` is a document-level compatibility aggregate of page OCR text, matching the Backend integration report.
- Figure/Table and Stamp/Signature store one canonical JSON-as-TXT file per page.
- Figure/Table and Stamp/Signature additionally store one deterministic PNG crop per detected object.
- Crops are generated from the EXIF-corrected source image using canonical source-coordinate bounding boxes.
- Reprocessing cleans only the three AI-owned module prefixes and rewrites deterministic names. Backend-owned `original.pdf`, `main.txt`, and `images/` are never modified.

MinIO has object-key prefixes rather than real directories; explicit folder creation is not required.

## Product response

Success:

```json
{
  "document_id": "123",
  "status": "success"
}
```

Model-processing failure uses a non-2xx HTTP status and returns:

```json
{
  "document_id": "123",
  "status": "failed",
  "error": "..."
}
```

HTTP semantics:

- `200`: all required model processing and artifact persistence succeeded;
- `404`: input MinIO object does not exist;
- `422`: invalid request, URL, image, or document identifier;
- `500`: one or more model pipelines failed;
- `502`: MinIO read/write/connectivity failure;
- `503`: MinIO disabled or misconfigured.

## Internal vs public MinIO addresses

Two addresses remain intentionally separate:

- `WIKI_HAMI_MINIO_ENDPOINT`: address used by Wiki Hami's MinIO SDK.
- `WIKI_HAMI_MINIO_PUBLIC_BASE_URL`: host/port expected in image URLs supplied to the API and used when canonical object URLs are built.

For the current server contract, Backend sends `http://minio:9000/media/...`, so the expected server configuration is:

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=minio:9000
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://minio:9000
WIKI_HAMI_MINIO_BUCKET=media
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
```

For local development against the exposed MinIO S3 port:

```env
WIKI_HAMI_MINIO_ENABLED=true
WIKI_HAMI_MINIO_ENDPOINT=192.168.4.209:9002
WIKI_HAMI_MINIO_PUBLIC_BASE_URL=http://192.168.4.209:9002
WIKI_HAMI_MINIO_BUCKET=media
WIKI_HAMI_MINIO_ACCESS_KEY=<secret>
WIKI_HAMI_MINIO_SECRET_KEY=<secret>
WIKI_HAMI_MINIO_SECURE=false
```

The MinIO Console port is not used by the AI service.

The AI MinIO credential now requires write permission as well as read/list access. At minimum it needs the equivalent of `ListBucket`, `GetObject`, and `PutObject`; `DeleteObject` is needed for retry cleanup of AI-owned prefixes.

## Security

Backend-provided image URLs are never fetched with an arbitrary HTTP client. Wiki Hami parses the URL and requires:

- `http` or `https`;
- host/port matching `WIKI_HAMI_MINIO_PUBLIC_BASE_URL`;
- bucket matching `WIKI_HAMI_MINIO_BUCKET`;
- a non-empty object key;
- no embedded credentials, fragments, `.` or `..` path segments.

After validation, bucket/object identity is read through the authenticated MinIO SDK. Query strings may be accepted for object identity, but transient query credentials are never persisted in provenance.

## Engineering/local endpoints

The module endpoints remain available for isolated model debugging/evaluation:

```text
POST /api/v1/ocr
POST /api/v1/figure-table
POST /api/v1/stamp-signature
```

They return `ModulePageResponse` and do not publish product artifacts.

Detailed full-document inspection paths are:

```text
POST /api/v1/extract
POST /api/v1/extract/minio/inspect
```

Both return the existing `DocumentExtractionResponse`. They share the same preprocessing/model orchestration as production. For product-like local testing, they support optional artifact persistence without changing their detailed response:

- multipart `/extract`: form field `persist_outputs=true`;
- `/extract/minio/inspect`: query parameter `persist_outputs=true`.

The Streamlit inspector exposes this as **Persist product artifacts to MinIO**.

## Storage inspection endpoints

Internal Streamlit/dev storage helpers remain:

```text
GET /api/v1/storage/minio/health
GET /api/v1/storage/minio/objects?prefix=...
GET /api/v1/storage/minio/object?object_key=...
```

Object listing/preview are controlled by `WIKI_HAMI_MINIO_BROWSER_ENABLED` and can be disabled in deployments that do not expose the inspection UI.
