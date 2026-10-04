from fastapi import APIRouter

from app.core.config import get_settings
from app.modules.ocr.backend import create_ocr_backend

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get(
    "/health",
    summary="Service/configuration liveness",
    description=(
        "Reports configured model backends and MinIO settings without loading models or contacting "
        "MinIO/Ollama. Use /api/v1/storage/minio/health when storage connectivity must be verified."
    ),
)
async def health():
    ocr_metadata = create_ocr_backend(settings).metadata
    return {
        "status": "ok",
        "service": settings.app_name,
        "schema_version": settings.schema_version,
        "modules": {
            "ocr": {
                "mode": settings.text_extraction_mode,
                "backend": ocr_metadata.backend,
                "model_id": ocr_metadata.model_id,
            },
            "figure_table": {
                "backend": settings.figure_table_backend,
                "model_id": settings.figure_table_model_id,
            },
            "stamp_signature": {
                "backend": settings.stamp_signature_backend,
                "model_id": settings.stamp_signature_model_id,
            },
        },
        "storage": {
            "minio": {
                "enabled": settings.minio_enabled,
                "endpoint": settings.minio_endpoint,
                "public_base_url": settings.minio_public_base_url,
                "bucket": settings.minio_bucket,
                "browser_enabled": settings.minio_browser_enabled,
            }
        },
    }
