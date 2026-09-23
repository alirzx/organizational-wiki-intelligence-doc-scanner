from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.core.config import get_settings

settings = get_settings()

TAGS_METADATA = [
    {
        "name": "OCR",
        "description": "Engineering OCR endpoint for one MinIO image. Production document processing uses /extract/minio.",
    },
    {
        "name": "Figure / Table",
        "description": "Engineering PP-DocLayout endpoint for one MinIO image. Production document processing uses /extract/minio.",
    },
    {
        "name": "Stamp / Signature",
        "description": "Engineering RF-DETR endpoint for one MinIO image. Production document processing uses /extract/minio.",
    },
    {
        "name": "Full Extraction",
        "description": "Production MinIO document workflow plus local/debug full-extraction paths.",
    },
    {
        "name": "MinIO / Dev Storage",
        "description": "MinIO connectivity, object listing, and preview endpoints used by internal inspection tooling.",
    },
    {"name": "Health", "description": "Process and configuration liveness."},
]

app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description=(
        "Wiki Hami Extraction V1. The primary product integration is POST /api/v1/extract/minio: "
        "the backend sends all MinIO-backed page images for one document, Wiki Hami runs OCR, "
        "figure/table and stamp/signature extraction, persists AI-owned artifacts back to MinIO, "
        "and returns a small success/failed status response. Local multipart upload and detailed "
        "MinIO inspection remain available for engineering and Streamlit workflows."
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
    }
