# Extraction V1 Deployment Runbook

## Production topology

The root Compose project is self-contained for model serving while Redis, MinIO,
Backend, and the external `wikio` network remain shared infrastructure.

```text
Backend
   |
   v
Wiki Hami API ----> Redis ----> Wiki Hami worker
   |                              |
   |                              +--> OCR / layout / stamp models
   |                              +--> MinIO artifacts
   |                              +--> Backend callback
   |
   +--> isolated Ollama <---- persistent project model volume
             |
             +--> deepseek-ocr:latest (when VLM mode is selected)
```

Services are `api`, `worker`, optional `ui`, project-local `ollama`, and
one-shot `ollama-model-init`. Backend request/callback and MinIO artifact
contracts are unchanged.

## Isolated Ollama lifecycle

Ollama stores its state at `/root/.ollama` in the named volume
`wiki_hami_ollama_models`. The bootstrap service exits immediately in classic
OCR mode. In VLM mode it waits for Ollama, checks `ollama show`, and pulls
`WIKI_HAMI_VLM_MODEL_ID` only when absent.

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=vlm
WIKI_HAMI_VLM_BACKEND=ollama
WIKI_HAMI_VLM_MODEL_ID=deepseek-ocr:latest
WIKI_HAMI_VLM_BASE_URL=http://ollama:11434
```

The Ollama port is exposed only on the Compose network.

### Memory and disk safety

Stage defaults apply cgroup limits instead of trusting the model runtime to consume
whatever the host has available. Ollama is capped at 10 GB RAM with swap disabled,
worker at 10 GB, API at 2 GB, and UI at 1 GB. Ollama serves one request at a time,
keeps one model loaded, queues at most four requests, and the VLM request uses a
30-second keep-alive so memory is released shortly after an OCR burst.

Docker json-file logs rotate via `WIKI_HAMI_LOG_MAX_SIZE` and
`WIKI_HAMI_LOG_MAX_FILES`; VLM diagnostic JSON is bounded by
`WIKI_HAMI_VLM_DIAGNOSTICS_MAX_FILES`. The project-owned Ollama volume is mounted
into the one-shot bootstrap for size checks; `WIKI_HAMI_OLLAMA_MODEL_STORE_MAX_MB`
defaults to 12288 MB and bootstrap fails if the store is already over the cap or
crosses it after a pull. This prevents silent model-cache growth, while operators
should still monitor the Docker data-root filesystem because filesystem-level
free space is shared with other Docker projects.

## OCR backend selection

DeepSeek/Ollama:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=vlm
```

PaddleOCR:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=paddle
```

Bina Rizeh:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=bina_rizeh
```

Switching backend does not change API, callback, MinIO, `OCR.txt`, or
`layout.json` contracts.

## CPU/GPU configuration

`WIKI_HAMI_MODEL_RUNTIME=gpu` builds CUDA Paddle/Torch packages. Use it whenever
any in-process OCR/Figure-Table/Stamp-Signature module uses a GPU. Individual
module devices can still be `cpu`.

For a fully CPU scanner:

```env
WIKI_HAMI_MODEL_RUNTIME=cpu
WIKI_HAMI_CONTAINER_RUNTIME=runc
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=none
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

GPU scanner container exposure:

```env
WIKI_HAMI_CONTAINER_RUNTIME=nvidia
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=all
```

Per-module GPU device syntax:

```env
WIKI_HAMI_OCR_DEVICE=gpu:0
WIKI_HAMI_FIGURE_TABLE_DEVICE=gpu:0
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cuda:0
```

Ollama CPU/GPU is independent:

```env
# GPU
WIKI_HAMI_OLLAMA_CONTAINER_RUNTIME=nvidia
WIKI_HAMI_OLLAMA_NVIDIA_VISIBLE_DEVICES=all

# CPU
# WIKI_HAMI_OLLAMA_CONTAINER_RUNTIME=runc
# WIKI_HAMI_OLLAMA_NVIDIA_VISIBLE_DEVICES=none
```

## Framework profiles

GPU build:

```text
PaddlePaddle GPU 3.2.2 (cu129 channel)
torch 2.14.0+cu130
torchvision 0.29.0+cu130
```

CPU build:

```text
paddlepaddle 3.2.2
torch 2.14.0+cpu
torchvision 0.29.0+cpu
```

## Start / update

```bash
cp .env.example .env
docker compose config
docker compose up -d --build
docker compose ps
```

VLM checks:

```bash
docker compose logs ollama-model-init
docker compose exec -T ollama ollama list
```

Runtime check:

```bash
docker compose exec -T worker python - <<'PY'
from app.core.config import Settings
s = Settings()
print("runtime:", s.model_runtime)
print("text mode:", s.text_extraction_mode)
print("ocr:", s.ocr_backend, s.ocr_device)
print("figure/table:", s.figure_table_device)
print("stamp/signature:", s.stamp_signature_device)
print("vlm:", s.vlm_backend, s.vlm_base_url, s.vlm_model_id)
PY
```

Keep the existing MinIO, Redis, API-key, and callback values from
`.env.example`. The project still joins the external `wikio` network.

## Acceptance test

Verify API/worker/Ollama state, worker queue connectivity, MinIO connectivity,
one real document through `queued -> processing -> completed`, published
per-page artifacts plus `OCR.txt` and `layout.json`, callback delivery, and
Backend `ready` state.

## Security / operations

- Never commit `.env` or secrets.
- Ollama is internal-only by default.
- Persistent model volumes should be monitored for disk use.
- Keep worker concurrency conservative for model memory.
- The GitLab CI definition is intentionally unchanged by this refactor.
