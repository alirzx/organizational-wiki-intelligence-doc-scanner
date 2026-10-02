from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get(
    "/health",
    summary="Service/configuration liveness",
    description=(
        "Reports configured model backends and MinIO settings without loading models or contacting "
        "MinIO. Use /api/v1/storage/minio/health when storage connectivity must be verified."
    ),
)
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "schema_version": settings.schema_version,
        "grouping": {
            "enabled": settings.grouping_enabled,
            "backend": settings.grouping_backend,
            "requires_trained_model": settings.grouping_backend == "lightgbm",
        },
        "modules": {
            "ocr": {"backend": settings.ocr_backend, "model_id": settings.ocr_model_id},
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
