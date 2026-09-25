# Models and Inference

This document describes the production model/runtime contract for Extraction V1. Model quality benchmarking and later semantic processing are outside this repository.

## Production model matrix

| Module | Backend setting | Baseline model | Default threshold | Canonical output |
|---|---|---|---:|---|
| OCR | `paddle` | `PP-OCRv5_server_det` + `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` | 0.45 | `paragraph` |
| Figure/Table | `pp_doclayout` | `PaddlePaddle/PP-DocLayoutV3` | 0.45 | `figure`, `table` |
| Stamp/Signature | `rfdetr` | `bluecopa/rf-detr-stamp-signature-detector` | 0.50 | `stamp`, `signature` |

Production is currently CPU-oriented. The `.env.example` defaults to mock backends so development/tests can run without loading model frameworks.

## Runtime lifecycle

Heavy model objects are loaded lazily on first inference and cached inside the process. Each real backend uses initialization/prediction locking around its shared model instance.

In production, model inference runs in the Celery worker. Current worker concurrency is `1`, avoiding uncontrolled duplication/parallel use of memory-heavy process-local model sets.

```text
fresh persistent cache + first inference
    -> load/download weights
    -> initialize model in worker RAM

same worker, later jobs
    -> reuse in-memory model

new worker process
    -> reuse persistent weights
    -> initialize another model instance in RAM
```

Increasing worker process count therefore multiplies model memory use.

## Shared preprocessing boundary

All modules receive the same `PreparedPage` created after:

- byte/image validation;
- Pillow decode;
- EXIF transpose;
- RGB conversion;
- source geometry capture;
- bounded shared long-edge resize.

Model-specific tensor normalization, padding, architecture-specific resize, etc. remain inside each backend.

Adapters restore detected geometry back to canonical source coordinates before creating `DetectedObject`.

Public coordinate space:

```text
exif_corrected_source_pixels
```

## OCR / PaddleOCR

Main files:

```text
app/modules/ocr/paddle_backend.py
app/modules/ocr/paragraph_grouper.py
app/modules/ocr/adapter.py
```

Baseline components:

- detector: `PP-OCRv5_server_det`;
- recognizer: `arabic_PP-OCRv5_mobile_rec`;
- optional text-line orientation model: `PP-LCNet_x1_0_textline_ori` when enabled;
- device: CPU by default.

Paddle produces text lines. Wiki Hami groups these lines geometrically into canonical paragraph objects. Paragraph grouping is intentionally not semantic section reconstruction.

Canonical paragraph contains source-space `bbox`, optional polygon, confidence, normalized `text`, `raw_text`, line/model metadata, and OCR provenance.

Current paragraph polygon is derived from the paragraph union bounding box, so it is rectangular rather than a richer text contour.

### OCR limitations

- grouping is geometry-based rather than semantic;
- multi-column/complex reading order is not fully modeled;
- no language-routing layer across multiple recognizers;
- no text correction/LLM cleanup in Extraction V1;
- OCR paragraph crops are not persisted.

## Figure/Table / PP-DocLayoutV3

Main files:

```text
app/modules/figure_table/pp_doclayout_backend.py
app/modules/figure_table/adapter.py
```

Backend parses model box predictions and maps selected labels to public object types.

Current mapping behavior:

- configured/exact table labels -> `table`;
- figure/image/chart-style labels -> `figure`;
- captions/titles are intentionally excluded from public Figure/Table V1 objects.

Metadata retains original model label/class ID.

Current PP-DocLayout geometry is parsed as XYXY box and exposed with a rectangular polygon derived from that box.

### Figure/Table limitations

- no cell/row/column extraction;
- no caption/title association;
- no semantic reading-order prediction;
- current polygon does not add shape precision beyond bbox.

## Stamp/Signature / RF-DETR

Main files:

```text
app/modules/stamp_signature/rfdetr_backend.py
app/modules/stamp_signature/adapter.py
```

Baseline model:

```text
bluecopa/rf-detr-stamp-signature-detector
```

Pinned checkpoint revision/config lives in settings. Public Extraction V1 behavior exposes only:

```text
stamp
signature
```

Non-target classes such as checkbox states are filtered out.

RF-DETR currently produces/restores source-space XYXY boxes; the canonical stamp/signature `polygon` is therefore `null`.

### Stamp/Signature limitations

- no signature identity;
- no stamp text/semantics interpretation;
- no overlap deduplication with OCR/other layout objects;
- no polygon/segmentation contour with the current backend.

## Unified layout interaction

All canonical objects from these modules are merged into document-level `layout.json` v2.

```text
paragraph + table + figure + stamp + signature
    -> page objects
    -> source geometry + provenance/artifact references
    -> deterministic geometric reading_order
```

The model services themselves do not assign the final unified reading order. `ArtifactPublisher` performs that merge/order when publishing the document layout.

## Mock backends

Mock backends are deterministic development/test fixtures. They validate orchestration, geometry restoration, artifacts, layout merging, and APIs without network/model downloads.

Mock detections are not model-quality evidence.

## Model caches

Framework caches should remain persistent across container recreation and outside Git.

Relevant environment locations:

```text
HF_HOME
PADDLE_HOME
PADDLE_PDX_CACHE_HOME
WIKI_HAMI_STAMP_SIGNATURE_CACHE_DIR (optional RF-DETR override)
```

Root Compose mounts a shared cache volume. Production-style Compose maps a persistent host data root.

Downloaded public weights should not be copied into the repository merely to avoid redownload; use persistent framework caches.

## First-run behavior

The API health endpoint does not load models. Model initialization occurs when the worker handles its first relevant extraction job (or when an engineering endpoint is invoked in the API process).

A cold environment may therefore have significantly higher first-job latency due to model download/initialization.

## Operational warnings

Warnings about deprecated Torch/RF-DETR APIs or model optimization should be evaluated separately from extraction failures. They do not by themselves mean a job failed.

The current RF-DETR runtime may report that the checkpoint class count differs from configured `num_classes`; the library uses the checkpoint class count. This warning should eventually be cleaned up in model configuration, but production correctness must be judged from actual mapped output/regression tests rather than the warning alone.

Native Paddle/runtime exceptions should be diagnosed from the page/module traceback and reproducible conditions. Do not assume every native failure is caused by page concurrency without evidence.

## Production configuration reference

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_MODEL_ID=PaddlePaddle/arabic_PP-OCRv5_mobile_rec
WIKI_HAMI_OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_server_det
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_OCR_SCORE_THRESHOLD=0.45
WIKI_HAMI_OCR_USE_TEXTLINE_ORIENTATION=true

WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_MODEL_ID=PaddlePaddle/PP-DocLayoutV3
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_SCORE_THRESHOLD=0.45

WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_MODEL_ID=bluecopa/rf-detr-stamp-signature-detector
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_SCORE_THRESHOLD=0.50
```

For exact default values, treat `app/core/config.py` and `.env.example` as the implementation source of truth.
