# Extraction V1 Contract Notes

## Coordinate space

Public detections use `exif_corrected_source_pixels`: the source image after EXIF display orientation is applied and before any model resize. Model backends may work in any internal coordinate space, but adapters must restore geometry before returning canonical objects.

Visual product crops are generated from `PreparedPage.source_image` using these canonical source-coordinate bounding boxes.

## Document identity

One logical document has one `document_id` and one or more page images. Each page has a stable `page_id`, `page_number`, and arbitrary page metadata. Every detected object repeats `document_id`, `page_id`, and `page_number` so flattened document-level results never lose provenance.

For product persistence, `document_id` must also be safe as a single MinIO object-key path segment because outputs are written under:

```text
documents/{document_id}/
```

## Product `/extract/minio` request

The primary Backend → AI request contains the whole document page list:

```json
{
  "document_id": "123",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
      "page_id": "123:p1",
      "page_number": 1
    }
  ]
}
```

The AI service reads all input images from the configured MinIO bucket and does not accept arbitrary remote HTTP images.

## Product output ownership

Wiki Hami owns only:

```text
documents/{document_id}/OCR/
documents/{document_id}/Figure-Table/
documents/{document_id}/Stamp-Signature/
documents/{document_id}/OCR.txt
```

Backend-owned source objects such as `images/`, `main.txt`, and `original.pdf` are outside the AI deletion/write boundary except for reading the page images supplied by Backend.

Per-page rules:

- OCR: one canonical JSON-as-TXT file and one plain extracted-text file; no image crops.
- Figure/Table: one canonical JSON-as-TXT file plus zero or more deterministic PNG crops.
- Stamp/Signature: one canonical JSON-as-TXT file plus zero or more deterministic PNG crops.

`OCR.txt` is a document-level aggregate compatibility artifact.

## Product response/failure boundary

`POST /api/v1/extract/minio` returns success only after all three modules succeed for all pages and all required artifact writes complete:

```json
{
  "document_id": "123",
  "status": "success"
}
```

Failures use non-2xx HTTP status and include `status: failed` plus an error message where the endpoint handles the failure directly. A partial module run is not published as a successful product document.

This is intentionally stricter than detailed engineering extraction.

## `/extract` multipart fields

- `document_id`: required string
- `images`: repeated file field, 1..N
- `document_metadata_json`: optional JSON object
- `pages_metadata_json`: optional JSON array, exactly one descriptor per image
- `persist_outputs`: optional boolean, default false

Example page descriptor:

```json
{
  "page_id": "doc_42:p7",
  "page_number": 7,
  "filename": "scan_0007.png",
  "metadata": {"source_asset_id": "asset_991"}
}
```

`persist_outputs=true` uses the product artifact publisher but retains the detailed local response.

## Detailed MinIO inspection

`POST /api/v1/extract/minio/inspect` accepts the same JSON body as product `/extract/minio`, returns the detailed `DocumentExtractionResponse`, and optionally accepts query parameter `persist_outputs=true` for one-pass product-like testing.

## Shared vs model-specific normalization

Shared preprocessing owns validation, decode, display orientation, color space, resizing, and geometry transforms. Mean/std normalization, tensor layout, tokenization, and model-specific resize/padding belong inside each backend adapter.

## Detailed status/failure boundary

Detailed `/extract` and `/extract/minio/inspect` preserve successful module results when another module fails or times out and can report `success`, `partial_success`, or `failed` in page/document `processing`. Upload, JSON, descriptor and image validation happen before inference and are all-or-nothing.

When detailed routes are asked to `persist_outputs=true`, full success is required before publishing so local product simulation follows the same persistence guarantee as production.
