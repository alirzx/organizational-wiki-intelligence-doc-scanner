# Extraction V1 Architecture Graph

This is the visual companion to [architecture.md](architecture.md). It reflects the
asynchronous production path and the current `app/` module boundaries.

## Production runtime

```mermaid
flowchart LR
    user[User] --> frontend[Frontend]
    frontend --> backend[Backend / Django + Celery]

    backend -->|write source PDF, text, and page images| minio[(MinIO)]
    backend -->|POST /api/v1/extract/minio<br/>X-API-Key| api[Wiki Hami FastAPI]

    api -->|create queued job| jobstore[(Redis DB 2<br/>job state)]
    api -->|enqueue task| broker[(Redis DB 0<br/>Celery broker)]
    broker --> worker[Celery worker<br/>concurrency 1]
    worker -.->|task results| results[(Redis DB 1<br/>Celery results)]

    worker -->|read page images| minio
    worker --> preprocess[Shared preprocessing]
    preprocess --> orchestrator[ExtractionOrchestrator]

    orchestrator --> ocr[OCR service<br/>PaddleOCR]
    orchestrator --> layout[Figure/Table service<br/>PP-DocLayoutV3]
    orchestrator --> marks[Stamp/Signature service<br/>RF-DETR]

    ocr --> merge[Canonical page/document result]
    layout --> merge
    marks --> merge
    merge --> publisher[ArtifactPublisher]
    publisher -->|write AI-owned artifacts| minio
    worker -->|processing / completed / failed| jobstore
    worker -->|terminal callback<br/>Bearer token| backend

    minio --> artifacts["OCR/ · Figure-Table/ · Stamp-Signature/<br/>OCR.txt · layout.json v2"]
```

The API returns `202 Accepted` after queueing. Model inference and artifact
publication happen in the worker. A callback failure does not change an otherwise
completed extraction to failed; the Backend can reconcile through the job-status API.

## Internal code map

```mermaid
flowchart TB
    entry[app/main.py] --> router[app/api/v1/router.py]

    router --> health[health endpoint]
    router --> storage_api[storage endpoints]
    router --> engineering[engineering extraction endpoints]
    router --> production[production extraction endpoint]
    router --> jobs_api[job-status endpoint]

    production --> auth[app/security.py]
    production --> jobs[app/jobs/tasks.py]
    jobs --> processor[app/jobs/processor.py]
    jobs --> store[app/jobs/store.py]
    jobs --> callback[app/jobs/callback.py]

    engineering --> parsing[app/api/v1/request_parsing.py]
    processor --> parsing
    parsing --> storage[app/storage/minio_service.py]
    parsing --> prep[app/preprocessing/pipeline.py]

    processor --> orchestration[app/orchestration/extractor.py]
    engineering --> orchestration

    orchestration --> ocr_service[app/modules/ocr/service.py]
    orchestration --> layout_service[app/modules/figure_table/service.py]
    orchestration --> mark_service[app/modules/stamp_signature/service.py]

    ocr_service --> paddle[PaddleOCR backend]
    ocr_service --> paragraph[paragraph grouper + adapter]
    layout_service --> ppdoc[PP-DocLayout backend + adapter]
    mark_service --> rfdetr[RF-DETR backend + adapter]

    paragraph --> schemas[app/schemas]
    ppdoc --> schemas
    rfdetr --> schemas
    prep --> schemas

    processor --> publisher[app/artifacts/publisher.py]
    publisher --> storage
    publisher --> schemas

    runtime[app/core/runtime.py<br/>process-local cached services] -.-> storage
    runtime -.-> store
    runtime -.-> publisher
    runtime -.-> orchestration
```

Solid arrows show request/data-flow dependencies. Dotted arrows show the
process-local service registry constructing and caching shared components.

## Artifact ownership

```mermaid
flowchart LR
    backend[Backend-owned] --> source["documents/{id}/<br/>original.pdf<br/>main.txt<br/>images/"]
    ai[AI-owned] --> output["documents/{id}/<br/>OCR/<br/>Figure-Table/<br/>Stamp-Signature/<br/>OCR.txt<br/>layout.json"]

    source -->|page input| pipeline[Extraction pipeline]
    pipeline -->|deterministic output keys| output
```

The pipeline may replace only AI-owned outputs. Public object geometry is expressed
in `exif_corrected_source_pixels`; adapters restore model-space detections to that
coordinate system before creating canonical detected objects.
