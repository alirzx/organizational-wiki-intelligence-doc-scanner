# Production Deployment

Root Compose runs API, worker, optional UI, and an isolated Ollama service.
Redis, MinIO, Backend, and network `wikio` remain shared infrastructure.

## Text extraction selection

VLM:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=vlm
WIKI_HAMI_VLM_MODEL_ID=deepseek-ocr:latest
WIKI_HAMI_VLM_BASE_URL=http://ollama:11434
```

Paddle:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=paddle
```

Bina:

```env
WIKI_HAMI_TEXT_EXTRACTION_MODE=ocr
WIKI_HAMI_OCR_BACKEND=bina_rizeh
```

## Runtime profiles

GPU scanner:

```env
WIKI_HAMI_MODEL_RUNTIME=gpu
WIKI_HAMI_CONTAINER_RUNTIME=nvidia
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=all
```

CPU scanner:

```env
WIKI_HAMI_MODEL_RUNTIME=cpu
WIKI_HAMI_CONTAINER_RUNTIME=runc
WIKI_HAMI_NVIDIA_VISIBLE_DEVICES=none
WIKI_HAMI_OCR_DEVICE=cpu
WIKI_HAMI_FIGURE_TABLE_DEVICE=cpu
WIKI_HAMI_STAMP_SIGNATURE_DEVICE=cpu
```

Ollama has independent CPU/GPU runtime variables in `.env.example`.

## Start / update

```bash
docker compose config
docker compose up -d --build
docker compose ps
```

The one-shot `ollama-model-init` service checks the persistent
`wiki_hami_ollama_models` volume and pulls the selected VLM model only when it
is absent.

Verify:

```bash
docker compose logs ollama-model-init
docker compose exec -T ollama ollama list
docker compose exec -T worker celery -A app.jobs.celery_app:celery_app inspect ping
```

Then submit one real Front/Backend document and confirm
`queued -> processing -> completed`, callback delivery, and expected MinIO
artifacts.

The GitLab CI definition is unchanged. It continues to copy the production env
file and run root Compose. The production env must therefore use
`WIKI_HAMI_VLM_BASE_URL=http://ollama:11434` when VLM mode is selected.

See [docs/deployment.md](../docs/deployment.md) for the full runbook.
