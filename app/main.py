from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.core.config import get_settings

settings = get_settings()

TAGS_METADATA = [
    {"name": "OCR", "description": "Engineering OCR endpoint for one MinIO image."},
    {"name": "Figure / Table", "description": "Engineering PP-DocLayout endpoint for one MinIO image."},
    {"name": "Stamp / Signature", "description": "Engineering RF-DETR endpoint for one MinIO image."},
    {"name": "Full Extraction", "description": "Async production MinIO workflow plus synchronous local/debug extraction."},
    {"name": "Async Jobs", "description": "Durable asynchronous extraction status and recovery."},
    {"name": "MinIO / Dev Storage", "description": "MinIO connectivity, object listing, and preview endpoints for internal inspection."},
    {"name": "Health", "description": "Process and configuration liveness."},
]

app = FastAPI(
    title=settings.app_name,
    version="0.4.0",
    description=(
        "Wiki Hami Extraction V1. POST /api/v1/extract/minio validates the existing document/page MinIO request, "
        "creates a durable asynchronous job and returns HTTP 202 + job_id. Celery workers run OCR, figure/table "
        "and stamp/signature extraction, persist OCR.txt, layout.json and per-page artifacts to MinIO, then send "
        "a terminal callback to Backend. GET /api/v1/jobs/{job_id} provides polling/recovery."
    ),
    openapi_tags=TAGS_METADATA,
)
app.include_router(v1_router, prefix=settings.api_prefix)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "minio_health": f"{settings.api_prefix}/storage/minio/health",
        "product_extract": f"{settings.api_prefix}/extract/minio",
        "job_status": f"{settings.api_prefix}/jobs/<job_id>",
    }
