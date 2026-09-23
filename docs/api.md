# Extraction V1 API reference

Base prefix: `/api/v1`. Interactive OpenAPI/Swagger documentation is available at `/docs`.

## Integration boundaries

Primary Backend → AI production integration:

```text
POST /api/v1/extract/minio
```

The Backend sends one document plus all prepared MinIO page images. Wiki Hami runs all three extraction modules, persists AI-owned outputs back to MinIO, and returns only document processing status.

Engineering/debug endpoints remain available:

```text
POST /api/v1/ocr
POST /api/v1/figure-table
POST /api/v1/stamp-signature
POST /api/v1/extract
POST /api/v1/extract/minio/inspect
```

These call the same in-process services as production. No endpoint performs loopback HTTP calls to another Wiki Hami endpoint.

## Product document request

`POST /api/v1/extract/minio` accepts JSON:

```json
{
  "document_id": "123",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
      "page_number": 1,
      "page_id": "123:p1"
    },
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-002.jpg",
      "page_number": 2,
      "page_id": "123:p2"
    }
  ]
}
```

Fields:

| Field | Required | Meaning |
|---|---:|---|
| `document_id` | yes | Logical document identity and output path segment |
| `document_metadata` | no | Arbitrary document metadata |
| `pages` | yes | Ordered list of one or more MinIO-backed pages |
| `pages[].image_url` | yes | MinIO object URL matching configured host/port and bucket policy |
| `pages[].page_number` | no | Integer >= 1; request order is used as fallback |
| `pages[].page_id` | no | Defaults to `<document_id>:p<page_number>` during preparation |
| `pages[].filename` | no | Optional filename override |
| `pages[].metadata` | no | Arbitrary per-page metadata |

The URL is parsed only to identify the configured bucket/object. Bytes are read through the authenticated MinIO SDK client. See [MinIO integration](minio.md).

## Product document response

Success:

```json
{
  "document_id": "123",
  "status": "success",
  "error": null
}
```

Processing/storage failure uses a non-2xx status and the contract shape:

```json
{
  "document_id": "123",
  "status": "failed",
  "error": "..."
}
```

The response intentionally excludes the full extraction payload because canonical raw results and crops are persisted in MinIO.

## Product artifact output

For each successful document, Wiki Hami writes:

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

There are no OCR crops. Visual-module crops use source-image bounding boxes. Filenames are deterministic so retries overwrite the same artifact identities. Before persistence, only the three AI-owned module prefixes are cleaned; Backend-owned `images/`, `main.txt`, and `original.pdf` are not touched.

## Engineering module request

The three isolated module endpoints accept one MinIO-backed page:

```json
{
  "document_id": "123",
  "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
  "page_number": 1,
  "page_id": "123:p1",
  "page_metadata": {}
}
```

### OCR

`POST /api/v1/ocr` performs Paddle text detection/recognition plus paragraph grouping. Successful objects use `type: paragraph`.

### Figure/Table

`POST /api/v1/figure-table` performs PP-DocLayoutV3 localization and exposes canonical `figure` and `table` objects.

### Stamp/Signature

`POST /api/v1/stamp-signature` performs RF-DETR inference and exposes canonical `stamp` and `signature` objects. Filtered/non-target classes are not returned.

All three return `ModulePageResponse`, including image/transform provenance, detected objects and module status. An empty detection set is still module success with the relevant `no_*_detected` warning.

## Local multipart full extraction

`POST /api/v1/extract` is the multipart local/development workflow.

Fields:

- repeated `images`;
- `document_id`;
- optional `document_metadata_json`;
- optional `pages_metadata_json`;
- optional `persist_outputs` boolean.

With `persist_outputs=false` it returns the detailed canonical result only. With `persist_outputs=true`, it also publishes the same MinIO product artifacts before returning that detailed result.

## Detailed MinIO inspection

`POST /api/v1/extract/minio/inspect` accepts the same JSON request as the product endpoint and returns the existing `DocumentExtractionResponse` for Streamlit/testing.

Optional query parameter:

```text
persist_outputs=true
```

When enabled, it publishes the same product artifacts without requiring a second inference pass.

`DocumentExtractionResponse` includes `pages[]`, flattened `objects[]`, object counts and document processing status. Module failures remain isolated and can produce `partial_success` when persistence is not requested.

## Storage/health endpoints

- `GET /api/v1/health`: process/config liveness; does not contact MinIO or load models.
- `GET /api/v1/storage/minio/health`: verifies configured bucket connectivity.
- `GET /api/v1/storage/minio/objects`: internal/dev object browser when enabled.
- `GET /api/v1/storage/minio/object`: internal/dev object proxy for Streamlit preview when enabled.

## HTTP error semantics

Product `/extract/minio`:

- `200`: all modules and required MinIO writes succeeded;
- `404`: referenced input object not found;
- `422`: invalid request, URL, image or document identifier;
- `500`: module processing failure;
- `502`: MinIO read/write/connectivity failure;
- `503`: MinIO disabled/misconfigured.

## Canonical detected object

Every `DetectedObject` includes `object_id`, `document_id`, `page_id`, `page_number`, `type`, source-coordinate `bbox`, optional polygon, confidence, optional OCR text/raw text, metadata and model provenance.

Object types are `paragraph`, `table`, `figure`, `stamp`, and `signature`. Public coordinates use `exif_corrected_source_pixels`.
