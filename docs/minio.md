# MinIO Integration and Artifact Contract

## Bucket and ownership

Default bucket:

```text
media
```

Backend-owned inputs:

```text
media/documents/{document_id}/
├── original.pdf
├── main.txt
└── images/
    ├── page-001.jpg
    └── ...
```

AI-owned outputs:

```text
media/documents/{document_id}/
├── OCR/
│   ├── page-001.json.txt
│   ├── page-001-text.txt
│   └── ...
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
├── OCR.txt
└── layout.json
```

MinIO prefixes are object-key prefixes, not real folders.

## Backend request input policy

Production Backend sends page URLs such as:

```text
http://minio:9000/media/documents/52/images/page-001.jpg
```

The AI service does not perform arbitrary HTTP downloads. It validates the URL against configured MinIO authority/bucket/path and reads the object through the authenticated MinIO SDK.

Expected page prefix:

```text
documents/{document_id}/images/
```

## Per-page module artifacts

### OCR

For every successfully processed page:

```text
OCR/page-NNN.json.txt
OCR/page-NNN-text.txt
```

`page-NNN.json.txt` contains a serialized `ModulePageResponse` with:

- schema/request/document/page identity;
- image metadata;
- transform metadata;
- `objects[]`;
- module status.

OCR `objects[]` are canonical `paragraph` objects and may contain:

- `object_id`;
- document/page provenance;
- `bbox`;
- optional `polygon`;
- confidence;
- normalized `text`;
- `raw_text`;
- OCR metadata;
- model provenance.

OCR does not persist paragraph crop images.

### Figure-Table

For every successfully processed page:

```text
Figure-Table/page-NNN.json.txt
```

Detected visual objects additionally produce deterministic crops:

```text
page-NNN-table-001.png
page-NNN-table-002.png
page-NNN-figure-001.png
...
```

The raw page JSON exists even when no table/figure is detected.

### Stamp-Signature

For every successfully processed page:

```text
Stamp-Signature/page-NNN.json.txt
```

Detected visual objects additionally produce:

```text
page-NNN-stamp-001.png
page-NNN-signature-001.png
...
```

The raw page JSON exists even when no stamp/signature is detected.

## Document OCR aggregate

```text
documents/{document_id}/OCR.txt
```

Document-level plain OCR text formed from page OCR text in page processing order. It is an AI-owned compatibility/convenience artifact and is overwritten on successful republish.

## Unified `layout.json` v2

Path:

```text
documents/{document_id}/layout.json
```

One file per document.

Schema version:

```text
wiki-hami.layout.v2
```

Purpose: provide one page-aware reconstruction index for all five canonical object types.

### Top-level structure

```json
{
  "schema_version": "wiki-hami.layout.v2",
  "document_id": "52",
  "document_metadata": {"source": "minio"},
  "page_count": 3,
  "object_count": 27,
  "object_counts": {
    "paragraph": 18,
    "table": 2,
    "figure": 3,
    "stamp": 2,
    "signature": 2
  },
  "pages": []
}
```

### Page structure

```json
{
  "page_id": "52:p1",
  "page_number": 1,
  "width": 2480,
  "height": 3508,
  "coordinate_space": "exif_corrected_source_pixels",
  "image": {
    "filename": "page-001.jpg",
    "mime_type": "image/jpeg",
    "source_width": 2480,
    "source_height": 3508,
    "processed_width": 1770,
    "processed_height": 2500,
    "source_coordinate_space": "exif_corrected_source_pixels",
    "source": {}
  },
  "transform": {
    "exif_orientation_applied": false,
    "scale_x": 0.7137,
    "scale_y": 0.7127,
    "model_input_color_space": "RGB",
    "notes": []
  },
  "reading_order_method": "bbox_top_to_bottom_then_left_to_right",
  "object_count": 9,
  "object_counts": {
    "paragraph": 5,
    "table": 1,
    "figure": 1,
    "stamp": 1,
    "signature": 1
  },
  "objects": []
}
```

### Unified object structure

Each layout object starts from the full canonical `DetectedObject` and adds module/order/artifact information.

```json
{
  "object_id": "table-...",
  "document_id": "52",
  "page_id": "52:p1",
  "page_number": 1,
  "type": "table",
  "bbox": {
    "x1": 150.0,
    "y1": 500.0,
    "x2": 2200.0,
    "y2": 1300.0
  },
  "polygon": {
    "points": [
      {"x": 150.0, "y": 500.0},
      {"x": 2200.0, "y": 500.0},
      {"x": 2200.0, "y": 1300.0},
      {"x": 150.0, "y": 1300.0}
    ]
  },
  "confidence": 0.94,
  "text": null,
  "raw_text": null,
  "metadata": {
    "source_label": "table",
    "source_class_id": 3
  },
  "provenance": {
    "module": "figure_table",
    "backend": "pp_doclayout",
    "model_id": "PaddlePaddle/PP-DocLayoutV3",
    "model_version": null
  },
  "module": "figure_table",
  "reading_order": 2,
  "artifacts": {
    "module_result": "documents/52/Figure-Table/page-001.json.txt",
    "plain_text": null,
    "crop": "documents/52/Figure-Table/page-001-table-001.png"
  }
}
```

### Object-type behavior

```text
paragraph       bbox=yes   polygon=optional/currently rectangular   text=yes   crop=no
figure          bbox=yes   polygon=optional/currently rectangular   text=no    crop=yes
table           bbox=yes   polygon=optional/currently rectangular   text=no    crop=yes
stamp           bbox=yes   polygon=null with current backend         text=no    crop=yes
signature       bbox=yes   polygon=null with current backend         text=no    crop=yes
```

The schema keeps `polygon` optional so richer future geometry can be added without changing the object shape.

## Reading order

`reading_order` is **per page** and is assigned after all five object types are merged.

Current deterministic sort key:

```text
bbox.y1
bbox.x1
bbox.y2
bbox.x2
type
object_id
```

Then order is assigned from `1..N`.

This is geometric ordering, not a semantic RTL/multi-column reading-order model. For preview/document reconstruction, geometry is authoritative. A stamp/signature may overlap text and must remain at its source coordinates even if its numeric order differs from a human reading sequence.

## BBox and polygon

`bbox` is an axis-aligned XYXY box in source-page pixels:

```json
{"x1":100,"y1":200,"x2":500,"y2":350}
```

`polygon` is a list of at least three source-space points.

Current OCR paragraph and PP-DocLayout polygons are rectangular polygons derived from their boxes, so they do not yet provide more shape precision than `bbox`. RF-DETR stamp/signature currently provides only `bbox`.

For current preview reconstruction, `bbox + page width/height` is sufficient. Preserve `polygon` because future backends may provide real rotated/irregular geometry.

## Artifact references

Every unified object contains:

```json
"artifacts": {
  "module_result": "...",
  "plain_text": "... or null",
  "crop": "... or null"
}
```

This lets downstream consumers resolve from layout object to raw module response and visual crop without guessing filenames.

## Retry and ownership safety

Before republishing, AI removes only:

```text
documents/{id}/OCR/
documents/{id}/Figure-Table/
documents/{id}/Stamp-Signature/
```

Then it rewrites module artifacts and overwrites root `OCR.txt` and `layout.json`.

AI never deletes or rewrites:

```text
original.pdf
main.txt
images/*
```

## Callback object keys

Callback/job output values are object keys inside bucket `media`:

```text
documents/52/layout.json
```

Do not prepend `media/` when passing the key to an SDK call that already receives bucket `media` separately.
