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

Stage defaults are sized for the shared ~30 GB / 7-core host, not for a dedicated
AI machine. Ollama is capped at 10 GB RAM and 2 CPUs, worker at 7 GB / 1 CPU,
API at 1 GB / 0.5 CPU, and UI at 512 MB / 0.25 CPU. Thus the long-running Scanner
services can consume at most about 3.75 CPU cores, leaving more than three cores
for Backend, Frontend, Redis, MinIO, Template Maker, Doc Linker, and the OS.
Swap is disabled inside the Scanner containers by setting memory+swap equal to
the memory limit. Ollama serves one request at a time, queues at most two, keeps
one model loaded, and the VLM request uses a 30-second keep-alive.

CPU quota alone is not sufficient for CPU inference: each DeepSeek request also
sets `num_thread=2`, matching the Ollama 2-CPU cgroup budget. In-process numerical
libraries are capped to one OpenMP/BLAS/MKL/NumExpr thread to prevent nested CPU
oversubscription in Paddle/Torch/numpy code.

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

For the `stage` branch the defaults are already fully CPU-only. The explicit
production values are:

```env
WIKI_HAMI_MODEL_RUNTIME=cpu
WIKI_HAMI_CONTAINER_RUNTIME=runc
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=none
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
WIKI_HAMI_OLLAMA_CONTAINER_RUNTIME=runc
WIKI_HAMI_OLLAMA_NVIDIA_VISIBLE_DEVICES=none
WIKI_HAMI_VLM_NUM_THREADS=2
WIKI_HAMI_OLLAMA_CPUS=2.0
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


## Shared-host resource verification

After deployment, verify the *effective* cgroup quotas instead of trusting the env file:

```bash
docker compose config | sed -n '/ollama:/,/ollama-model-init:/p'
docker inspect "$(docker compose ps -q ollama)" --format 'NanoCPUs={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}'
docker inspect "$(docker compose ps -q worker)" --format 'NanoCPUs={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}} MemorySwap={{.HostConfig.MemorySwap}}'
docker compose exec -T ollama sh -lc 'env | grep -E "OLLAMA_(NUM_PARALLEL|MAX_QUEUE|MAX_LOADED_MODELS|MAX_TRANSFER_STREAMS|CONTEXT_LENGTH)|GOMAXPROCS" | sort'
docker stats --no-stream
```

With no inference request and after the 30-second keep-alive expires, `ollama ps`
should show no loaded model and Ollama CPU usage should settle near idle. If CPU
remains high, inspect `docker compose logs --tail=200 ollama ollama-model-init`
before raising any CPU quota; startup download/model preparation is distinct from
steady-state idle behavior.


## CPU VLM resilience

Ollama chat responses are streamed. `WIKI_HAMI_VLM_TIMEOUT_SECONDS` therefore
measures socket inactivity, not total generation duration. CPU VLM OCR has a
separate page-module limit `WIKI_HAMI_VLM_MODULE_TIMEOUT_SECONDS=1200`, since
three region requests can execute sequentially.

Region generation is bounded by `WIKI_HAMI_VLM_REGION_MAX_TOKENS=2048`, while
whole-page recovery keeps `WIKI_HAMI_VLM_MAX_TOKENS=3072`. Transport failures
fall back to classic OCR when `WIKI_HAMI_VLM_CLASSIC_FALLBACK=true`.

The llama.cpp prompt cache is bounded by `WIKI_HAMI_OLLAMA_CACHE_RAM_MB=512`.
Queued/processing jobs can be cancelled through
`DELETE /api/v1/jobs/{job_id}`; cancellation uses the existing failed callback
shape with `error.code=CANCELLED`.
