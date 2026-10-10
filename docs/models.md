# Models and Inference

## Selection matrix

| Module | Selection | Backend/model | Runtime control |
|---|---|---|---|
| OCR | `TEXT_EXTRACTION_MODE=vlm` | project Ollama + `deepseek-ocr:latest` | Ollama container runtime |
| OCR | `TEXT_EXTRACTION_MODE=ocr`, `OCR_BACKEND=paddle` | PaddleOCR | `OCR_DEVICE` |
| OCR | `TEXT_EXTRACTION_MODE=ocr`, `OCR_BACKEND=bina_rizeh` | PP-OCRv6 detector + Bina | `OCR_DEVICE` |
| Figure/Table | `pp_doclayout` | PP-DocLayoutV3 | `FIGURE_TABLE_DEVICE` |
| Stamp/Signature | `rfdetr` | RF-DETR checkpoint | `STAMP_SIGNATURE_DEVICE` |

Backend/API/artifact contracts do not change with OCR selection.

## Project-local Ollama

Compose owns `ollama` at `http://ollama:11434` and a persistent
`wiki_hami_ollama_models` volume mounted at `/root/.ollama`. The init service
pulls only the selected `WIKI_HAMI_VLM_MODEL_ID` when it is missing and skips
pulling in classic OCR mode.

Ollama CPU/GPU is selected independently from scanner Paddle/Torch packages with
`WIKI_HAMI_OLLAMA_CONTAINER_RUNTIME` and
`WIKI_HAMI_OLLAMA_NVIDIA_VISIBLE_DEVICES`.

## Scanner dependency profiles

`WIKI_HAMI_MODEL_RUNTIME=gpu` installs the validated CUDA Paddle/Torch wheels.
Use it if any in-process module needs a GPU; other modules may still select CPU.

`WIKI_HAMI_MODEL_RUNTIME=cpu` installs CPU Paddle/Torch wheels and requires all
active in-process module devices to be `cpu`.

Files:

```text
requirements-paddle-cpu.txt
requirements-paddle-gpu.txt
requirements-torch-cpu.txt
requirements-torch-gpu.txt
requirements-models.txt
```

Device requests are explicitly validated. A requested GPU never silently falls
back to CPU.

## OCR behavior

Paddle and Bina keep bbox/polygon-aware line output and shared canonical paragraph
conversion. Bina uses the pinned PP-OCRv6 detector plus pinned Persian recognizer.

VLM uses the project Ollama service. Canonical VLM text removes model transport
artifacts and generated image-caption leakage. Dense Persian pages use the
top-to-bottom three-region strategy, then bounded full-page VLM recovery, then
Paddle fallback.

## Other modules

PP-DocLayout accepts `gpu:<index>` or `cpu`. The requested Paddle device is
validated before model creation.

RF-DETR accepts `cuda:<index>` or `cpu`. The requested Torch device is
validated before model creation.

## Persistent caches

```text
wiki_hami_model_cache   -> /app/.cache
wiki_hami_ollama_models -> /root/.ollama
```

Weights survive container recreation; model objects are still initialized per
process and can duplicate RAM/VRAM when process counts increase.

## Configuration examples

VLM GPU:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=vlm
WIKI_HAMI_VLM_MODEL_ID=deepseek-ocr:latest
WIKI_HAMI_VLM_BASE_URL=http://ollama:11434
WIKI_HAMI_OLLAMA_CONTAINER_RUNTIME=nvidia
WIKI_HAMI_OLLAMA_NVIDIA_VISIBLE_DEVICES=all
```

Paddle GPU:

```env
WIKI_HAMI_MODEL_RUNTIME=gpu
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=paddle
WIKI_HAMI_OCR_DEVICE=gpu:0
```

Bina CPU:

```env
WIKI_HAMI_MODEL_RUNTIME=cpu
WIKI_HAMI_CONTAINER_RUNTIME=runc
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=none
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=bina_rizeh
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

For exact defaults, use `app/core/config.py`, `.env.example`, and
`compose.yaml`.
