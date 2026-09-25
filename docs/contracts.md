# Extraction V1 Contract Notes

## Coordinate space

All public detections use:

```text
exif_corrected_source_pixels
```

This is the source image after EXIF display orientation and before shared inference resize. Model backends may use their own internal geometry, but adapters restore coordinates before creating canonical objects.

`BBox` uses XYXY source pixels:

```json
{"x1":100,"y1":200,"x2":500,"y2":350}
```

`Polygon` is optional and contains at least three source-space points. Current OCR paragraph and PP-DocLayout polygons are rectangular geometry derived from boxes; RF-DETR stamp/signature currently returns no polygon. Consumers should therefore treat `bbox` as sufficient for current preview placement while preserving `polygon` for future richer geometry.

## Document and page identity

One logical document has one safe `document_id` and one or more pages. Each page has stable `page_id` and `page_number`. Production request validation prevents duplicate page IDs/numbers.

Every canonical detected object repeats:

```text
document_id
page_id
page_number
```

so flattened or unified layouts do not lose source provenance.

## Canonical object types

```text
paragraph
table
figure
stamp
signature
```

Each `DetectedObject` contains:

```text
object_id
document_id
page_id
page_number
type
bbox
polygon?
confidence
text?
raw_text?
metadata
provenance
```

## Production request contract

```http
POST /api/v1/extract/minio
X-API-Key: <secret>
```

The request contains the complete document page list. AI validates that each image belongs under:

```text
documents/{document_id}/images/
```

The API creates a durable job and immediately returns `202 queued`. It does not perform production inference in the request thread.

## Job contract

States:

```text
queued -> processing -> completed
                    \-> failed
```

Recovery endpoint:

```http
GET /api/v1/jobs/{job_id}
```

Job-status success uses internal/recovery field `outputs`.

## Callback contract

Success callback uses:

```json
{
  "job_id": "...",
  "document_id": "...",
  "status": "completed",
  "result": {
    "ocr": "...",
    "layout": "...",
    "ocr_dir": "...",
    "figure_table_dir": "...",
    "stamp_signature_dir": "..."
  }
}
```

Failure callback uses:

```json
{
  "job_id": "...",
  "document_id": "...",
  "status": "failed",
  "error": {
    "code": "...",
    "message": "..."
  }
}
```

Do not conflate callback `result` with job-status `outputs`.

## Storage ownership contract

Backend owns:

```text
documents/{id}/original.pdf
documents/{id}/main.txt
documents/{id}/images/*
```

AI owns:

```text
documents/{id}/OCR/*
documents/{id}/Figure-Table/*
documents/{id}/Stamp-Signature/*
documents/{id}/OCR.txt
documents/{id}/layout.json
```

AI may delete only the three module prefixes during deterministic republish. Root AI artifacts are overwritten by key. Backend-owned inputs are never modified/deleted.

## Per-page artifact contract

OCR:

```text
OCR/page-NNN.json.txt
OCR/page-NNN-text.txt
```

No OCR crop images.

Figure/Table:

```text
Figure-Table/page-NNN.json.txt
Figure-Table/page-NNN-table-NNN.png
Figure-Table/page-NNN-figure-NNN.png
```

Stamp/Signature:

```text
Stamp-Signature/page-NNN.json.txt
Stamp-Signature/page-NNN-stamp-NNN.png
Stamp-Signature/page-NNN-signature-NNN.png
```

Visual crops are conditional on detections. Raw page JSON exists even for an empty successful detection set.

## Document-level artifact contract

```text
OCR.txt
layout.json
```

`OCR.txt` aggregates page OCR text.

`layout.json` is `wiki-hami.layout.v2` and contains all five canonical types per page, source page dimensions, geometry, confidence, metadata/provenance, artifact references, counts, and deterministic geometric reading order.

## Reading-order contract

Reading order is assigned independently per page after merging all five object types.

Current geometric sort:

```text
bbox.y1 -> bbox.x1 -> bbox.y2 -> bbox.x2 -> type -> object_id
```

then numbered `1..N`.

This is deterministic placement/sequence metadata, not semantic Persian/RTL or multi-column interpretation. Preview reconstruction must preserve `bbox`/`polygon` positions and may use reading order as an auxiliary sequence.

## Detailed engineering contracts

`POST /extract` and `POST /extract/minio/inspect` remain synchronous detailed inspection surfaces returning `DocumentExtractionResponse`.

They may expose `success`, `partial_success`, or `failed` processing detail. If asked to `persist_outputs=true`, full success is required before publishing product artifacts.

## Shared versus model-specific normalization

Shared preprocessing owns:

- byte/image validation;
- decode;
- EXIF display orientation;
- RGB conversion;
- bounded common resize;
- coordinate transforms.

Model-specific backends own:

- tensor layout;
- model normalization;
- model-specific padding/resizing;
- tokenization or architecture-specific preparation.

The public adapter boundary always returns canonical source-space objects.
