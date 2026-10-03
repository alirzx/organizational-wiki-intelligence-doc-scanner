# Models and Inference

This document describes the production model/runtime contract for Extraction V1. Model quality benchmarking and later semantic processing are outside this repository.

## Production model matrix

| Module | Backend setting | Baseline model | Default threshold | Canonical output |
|---|---|---|---:|---|
| OCR | `mock` | deterministic test line | — | `paragraph` |
| OCR | `paddle` | `PP-OCRv5_server_det` + `PaddlePaddle/arabic_PP-OCRv5_mobile_rec` | 0.45 | `paragraph` |
| OCR | `bina_rizeh` | independent `PaddlePaddle/PP-OCRv6_medium_det` + `Reza2kn/Bina-0.2-RizehPizeh` | 0.0 | `paragraph` |
| Figure/Table | `pp_doclayout` | `PaddlePaddle/PP-DocLayoutV3` | 0.45 | `figure`, `table` |
| Stamp/Signature | `rfdetr` | `bluecopa/rf-detr-stamp-signature-detector` | 0.50 | `stamp`, `signature` |

Model devices default to `gpu:0` for Paddle modules and `cuda:0` for RF-DETR.
The `.env.example` still defaults to mock backends so development/tests remain
model-free. Set all three device settings to `cpu` for the portable CPU profile.

## Pinned OCR runtime baseline

The OCR stack is intentionally pinned rather than allowed to float between rebuilds:

```text
paddlepaddle 3.2.2
paddleocr    3.7.0
paddlex      3.7.2
```

CPU and GPU framework pins live in separate requirements files. GPU installation
also requires operator-provided official Paddle and PyTorch indexes selected for
the verified target CUDA runtime; this repository does not invent that choice.
`requirements-models.txt` owns the shared PaddleOCR/PaddleX pins.

For CPU stability, Wiki Hami explicitly passes:

```text
enable_mkldnn = false
```

to `PaddleOCR` by default through `WIKI_HAMI_OCR_ENABLE_MKLDNN=false`. PaddleOCR/PaddleX normally enable the oneDNN/MKLDNN CPU path; the project keeps that optimization disabled unless the exact target runtime has been regression-tested. This is a runtime-stability choice, not a model-quality change.

## Runtime lifecycle

Heavy model objects are loaded lazily on first inference and cached inside the process. Each real backend uses initialization/prediction locking around its shared model instance.

Async production inference runs in the Celery worker. Synchronous engineering
endpoints can also infer in the API process, which is why both roles receive the
same model image, cache, and GPU reservation. Current worker concurrency is `1`.

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

All modules receive the same `PreparedPage` created after byte/image validation, Pillow decode, EXIF transpose, RGB conversion, source geometry capture, and bounded shared long-edge resize.

Model-specific tensor normalization, padding, architecture-specific resize, etc. remain inside each backend. Adapters restore detected geometry back to canonical source coordinates before creating `DetectedObject`.

Public coordinate space:

```text
exif_corrected_source_pixels
```

## OCR backends

Main files:

```text
app/modules/ocr/paddle_backend.py
app/modules/ocr/bina_rizeh_backend.py
app/modules/ocr/backend.py
app/modules/ocr/paragraph_grouper.py
app/modules/ocr/adapter.py
```

`WIKI_HAMI_OCR_BACKEND` is explicitly validated and supports only `mock`,
`paddle`, and `bina_rizeh`. It is a fixed in-process registry, not a mechanism
for dynamically importing arbitrary model code. All real backends return the
same model-space `OCRLine` representation, then the shared adapter groups lines
and restores public geometry to `exif_corrected_source_pixels`.

### Paddle fallback

Baseline components:

- detector: `PP-OCRv5_server_det`;
- recognizer: `arabic_PP-OCRv5_mobile_rec`;
- optional text-line orientation model: `PP-LCNet_x1_0_textline_ori` when enabled;
- device: `gpu:0` by default, with explicit `cpu` override;
- MKLDNN/oneDNN: disabled by default through `WIKI_HAMI_OCR_ENABLE_MKLDNN=false`.

Paddle produces text lines. Wiki Hami groups these lines geometrically into canonical paragraph objects. Paragraph grouping is intentionally not semantic section reconstruction.

Canonical paragraph contains source-space `bbox`, optional polygon, confidence, normalized `text`, `raw_text`, line/model metadata, and OCR provenance.

Current paragraph polygon is derived from the paragraph union bounding box, so it is rectangular rather than a richer text contour.

### OCR operational notes

Each real backend's shared model instances are guarded by a prediction lock.
The document orchestrator may keep multiple page pipelines active, but predictions
for one backend instance are serialized through that lock.

`WIKI_HAMI_MODULE_TIMEOUT_SECONDS` is a per-module/per-page orchestration timeout; it is not a whole-document Celery time limit. Native Paddle failures such as `RuntimeError: std::exception` are separate from that timeout and should be diagnosed from model/runtime logs and reproducibility.

### Bina Rizeh full-page backend

`bina_rizeh` is an in-process full-page composition, not the incompatible
combined `PaddleOCR` pipeline. It lazily creates an independent `TextDetection`
using `PaddlePaddle/PP-OCRv6_medium_det` revision
`8e0f56fb2ef86b461d99cfc7ac5c137738985f61`, rectifies each detected quadrilateral
into a horizontal line crop (rotating tall crops clockwise), and passes batches
to `TextRecognition` from `Reza2kn/Bina-0.2-RizehPizeh` revision
`993527413ff74ef6d446df91c715a4e0825abe5b`.

Only each repository's three runtime inference files are downloaded through the
Hugging Face cache. Files are validated before model initialization, including
Git LFS pointer rejection. Detector polygons remain in prepared-page coordinates
and the existing adapter restores them to source space exactly once. Wiki Hami
keeps logical text in `text` and visual-order recognizer output in `raw_text`.
Bina scores are model scores, not calibrated
probabilities, so `WIKI_HAMI_OCR_BINA_SCORE_THRESHOLD` is distinct from
Paddle's `WIKI_HAMI_OCR_SCORE_THRESHOLD`.

Bina targets Persian handwriting and printed text. Its model card does not claim
unrelated-language preservation or benchmarked English quality. This patch runs
one selected full-page backend per page; it does not add language detection or
automatic Persian/English per-line routing.

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

Backend parses model box predictions and maps selected labels to public object types. Configured/exact table labels map to `table`; figure/image/chart-style labels map to `figure`; captions/titles are intentionally excluded from public Figure/Table V1 objects. Metadata retains original model label/class ID.

Current PP-DocLayout geometry is parsed as XYXY box and exposed with a rectangular polygon derived from that box.

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

Pinned checkpoint revision/config lives in settings. Public Extraction V1 behavior exposes only `stamp` and `signature`; non-target classes such as checkbox states are filtered out. RF-DETR currently produces/restores source-space XYXY boxes, so the canonical stamp/signature `polygon` is `null`.

## Unified layout interaction

All canonical objects from these modules are merged into document-level `layout.json` v2.

```text
paragraph + table + figure + stamp + signature
    -> page objects
    -> source geometry + provenance/artifact references
    -> deterministic geometric reading_order
```

The model services themselves do not assign the final unified reading order. `ArtifactPublisher` performs that merge/order when publishing the document layout.

## Model caches

Framework caches should remain persistent across container recreation and outside Git.

Relevant environment locations:

```text
HF_HOME
PADDLE_HOME
PADDLE_PDX_CACHE_HOME
WIKI_HAMI_STAMP_SIGNATURE_CACHE_DIR
```

Root Compose mounts a shared cache volume. Production-style Compose maps a persistent host data root. Downloaded public weights should remain in framework caches rather than being copied into Git.

## Operational warnings

Warnings about deprecated Torch/RF-DETR APIs or model optimization should be evaluated separately from extraction failures. They do not by themselves mean a job failed.

The current RF-DETR runtime may report that the checkpoint class count differs from configured `num_classes`; the library uses the checkpoint class count. This is separate from OCR runtime stability.

## Production configuration reference

```env
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_MODEL_ID=PaddlePaddle/arabic_PP-OCRv5_mobile_rec
WIKI_HAMI_OCR_TEXT_DETECTION_MODEL_NAME=PP-OCRv5_server_det
WIKI_HAMI_OCR_DEVICE=gpu:0
WIKI_HAMI_OCR_ENABLE_MKLDNN=false
WIKI_HAMI_OCR_SCORE_THRESHOLD=0.45
WIKI_HAMI_OCR_USE_TEXTLINE_ORIENTATION=true

# Persian-focused Bina Rizeh full-page composition
WIKI_HAMI_OCR_BACKEND=bina_rizeh
WIKI_HAMI_OCR_BINA_MODEL_ID=Reza2kn/Bina-0.2-RizehPizeh
WIKI_HAMI_OCR_BINA_REVISION=993527413ff74ef6d446df91c715a4e0825abe5b
WIKI_HAMI_OCR_BINA_SCORE_THRESHOLD=0.0
WIKI_HAMI_OCR_BINA_BATCH_SIZE=1
WIKI_HAMI_OCR_DETECTION_MODEL_ID=PaddlePaddle/PP-OCRv6_medium_det
WIKI_HAMI_OCR_DETECTION_MODEL_REVISION=8e0f56fb2ef86b461d99cfc7ac5c137738985f61

WIKI_HAMI_FIGURE_TABLE_BACKEND=pp_doclayout
WIKI_HAMI_FIGURE_TABLE_MODEL_ID=PaddlePaddle/PP-DocLayoutV3
WIKI_HAMI_FIGURE_TABLE_DEVICE=gpu:0
WIKI_HAMI_FIGURE_TABLE_SCORE_THRESHOLD=0.45

WIKI_HAMI_STAMP_SIGNATURE_BACKEND=rfdetr
WIKI_HAMI_STAMP_SIGNATURE_MODEL_ID=bluecopa/rf-detr-stamp-signature-detector
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cuda:0
WIKI_HAMI_STAMP_SIGNATURE_SCORE_THRESHOLD=0.50
```

For exact default values, treat `app/core/config.py` and `.env.example` as the implementation source of truth.

Use `WIKI_HAMI_OCR_BACKEND=paddle` to roll back without changing API, artifact,
or layout contracts. Root and production Compose reserve an NVIDIA GPU for both
API and worker because synchronous engineering endpoints can infer in the API.
GPU builds require deployment-selected official Paddle/PyTorch indexes. GPU
execution remains unverified in this repository; the CPU profile remains available.
