import json
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import Settings
from app.preprocessing.pipeline import prepare_page
from app.preprocessing.validator import ImageValidationError
from app.schemas.image import ImageSourceMetadata, PageDescriptor
from app.storage.minio_service import (
    MinioConfigurationError,
    MinioObjectNotFound,
    MinioStorageError,
    MinioStorageService,
    MinioUrlError,
)
from app.utils.ids import default_page_id


def parse_json_object(raw: str | None, field_name: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON object")
    return value


def parse_page_descriptors(raw: str | None, count: int) -> list[PageDescriptor]:
    if not raw:
        return [PageDescriptor(page_number=i + 1) for i in range(count)]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="pages_metadata_json must be valid JSON") from exc
    if not isinstance(value, list) or len(value) != count:
        raise HTTPException(
            status_code=422,
            detail="pages_metadata_json must be an array with one entry per uploaded image",
        )
    try:
        return [PageDescriptor.model_validate(item) for item in value]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid page metadata: {exc}") from exc


def prepare_page_bytes(
    *,
    data: bytes,
    filename: str,
    mime_type: str | None,
    document_id: str,
    descriptor: PageDescriptor,
    fallback_page_number: int,
    settings: Settings,
    source: ImageSourceMetadata,
):
    page_number = descriptor.page_number or fallback_page_number
    page_id = descriptor.page_id or default_page_id(document_id, page_number)
    try:
        return prepare_page(
            data=data,
            filename=descriptor.filename or filename,
            mime_type=mime_type,
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
            page_metadata=descriptor.metadata,
            settings=settings,
            source=source,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "document_id": document_id,
                "page_id": page_id,
                "page_number": page_number,
                "error": str(exc),
            },
        ) from exc


async def prepare_uploaded_page(
    *,
    upload: UploadFile,
    document_id: str,
    descriptor: PageDescriptor,
    fallback_page_number: int,
    settings: Settings,
):
    data = await upload.read()
    return prepare_page_bytes(
        data=data,
        filename=upload.filename or f"page_{fallback_page_number}.image",
        mime_type=upload.content_type,
        document_id=document_id,
        descriptor=descriptor,
        fallback_page_number=fallback_page_number,
        settings=settings,
        source=ImageSourceMetadata(type="upload"),
    )


async def prepare_minio_page(
    *,
    image_url: str,
    document_id: str,
    descriptor: PageDescriptor,
    fallback_page_number: int,
    settings: Settings,
    storage: MinioStorageService,
):
    try:
        obj = await run_blocking(BlockingPool.IO, storage.fetch_url, image_url)
    except MinioUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MinioObjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MinioConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MinioStorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return prepare_page_bytes(
        data=obj.data,
        filename=obj.filename,
        mime_type=obj.content_type,
        document_id=document_id,
        descriptor=descriptor,
        fallback_page_number=fallback_page_number,
        settings=settings,
        source=ImageSourceMetadata(
            type="minio",
            url=obj.ref.source_url,
            bucket=obj.ref.bucket,
            object_key=obj.ref.object_key,
            etag=obj.etag,
        ),
    )
