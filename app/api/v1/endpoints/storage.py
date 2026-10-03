from fastapi import APIRouter, HTTPException, Query, Response

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import get_settings
from app.core.runtime import get_minio_storage_service
from app.schemas.storage import MinioHealthResponse, MinioObjectListResponse
from app.storage.minio_service import (
    MinioConfigurationError,
    MinioObjectNotFound,
    MinioStorageError,
    MinioUrlError,
)

router = APIRouter(prefix="/storage/minio", tags=["MinIO / Dev Storage"])
settings = get_settings()
storage = get_minio_storage_service()


def _require_browser() -> None:
    if not settings.minio_browser_enabled:
        raise HTTPException(status_code=403, detail="MinIO browser endpoints are disabled")


def _storage_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MinioUrlError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, MinioObjectNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, MinioConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@router.get(
    "/health",
    response_model=MinioHealthResponse,
    summary="Check MinIO configuration and bucket connectivity",
)
async def minio_health():
    if not settings.minio_enabled:
        return MinioHealthResponse(
            status="disabled",
            enabled=False,
            connected=False,
            endpoint=settings.minio_endpoint,
            public_base_url=settings.minio_public_base_url,
            bucket=settings.minio_bucket,
            browser_enabled=settings.minio_browser_enabled,
        )
    try:
        connected = await run_blocking(BlockingPool.IO, storage.check_connection)
    except MinioStorageError as exc:
        raise _storage_http_error(exc) from exc
    if not connected:
        raise HTTPException(
            status_code=503,
            detail=f"configured MinIO bucket {settings.minio_bucket!r} does not exist",
        )
    return MinioHealthResponse(
        status="ok",
        enabled=True,
        connected=True,
        endpoint=settings.minio_endpoint,
        public_base_url=settings.minio_public_base_url,
        bucket=settings.minio_bucket,
        browser_enabled=settings.minio_browser_enabled,
    )


@router.get(
    "/objects",
    response_model=MinioObjectListResponse,
    summary="List objects for the local Streamlit MinIO browser",
    description="Internal development/inspection endpoint. It never returns MinIO credentials.",
)
async def list_minio_objects(
    prefix: str = Query(default="", description="Optional object-key prefix."),
    limit: int | None = Query(default=None, ge=1),
):
    _require_browser()
    try:
        objects, truncated = await run_blocking(
            BlockingPool.IO,
            storage.list_objects,
            prefix=prefix,
            limit=limit,
        )
    except MinioStorageError as exc:
        raise _storage_http_error(exc) from exc
    return MinioObjectListResponse(
        bucket=settings.minio_bucket,
        prefix=prefix,
        count=len(objects),
        truncated=truncated,
        objects=objects,
    )


@router.get(
    "/object",
    summary="Proxy one MinIO object for local preview/annotation",
    description=(
        "Internal development endpoint used by Streamlit. The object is read from the configured "
        "bucket through the API so storage credentials never enter the browser/UI process."
    ),
)
async def get_minio_object(
    object_key: str = Query(min_length=1),
):
    _require_browser()
    try:
        obj = await run_blocking(BlockingPool.IO, storage.fetch_object, object_key)
    except MinioStorageError as exc:
        raise _storage_http_error(exc) from exc
    headers = {}
    if obj.etag:
        headers["ETag"] = obj.etag
    return Response(
        content=obj.data,
        media_type=obj.content_type or "application/octet-stream",
        headers=headers,
    )
