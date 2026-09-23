import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.api.v1.request_parsing import (
    parse_json_object,
    parse_page_descriptors,
    prepare_minio_page,
    prepare_uploaded_page,
)
from app.core.config import get_settings
from app.core.runtime import (
    get_artifact_publisher,
    get_extraction_orchestrator,
    get_minio_storage_service,
)
from app.schemas.extraction import DocumentExtractionResponse
from app.schemas.image import PageDescriptor
from app.schemas.status import ProcessingState
from app.schemas.storage import ExtractionJobResponse, MinioDocumentRequest
from app.storage.minio_service import (
    MinioConfigurationError,
    MinioStorageError,
    MinioUrlError,
)
from app.utils.ids import new_request_id

router = APIRouter(tags=["Full Extraction"])
settings = get_settings()
orchestrator = get_extraction_orchestrator()
storage = get_minio_storage_service()
publisher = get_artifact_publisher()


async def _prepare_minio_pages(payload: MinioDocumentRequest):
    if len(payload.pages) > settings.max_pages_per_document:
        raise HTTPException(
            status_code=422,
            detail=f"document exceeds max_pages_per_document={settings.max_pages_per_document}",
        )

    expected_prefix = f"documents/{payload.document_id}/images/"
    pages = []
    for index, item in enumerate(payload.pages, start=1):
        try:
            ref = storage.parse_image_url(item.image_url)
        except MinioUrlError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MinioConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not ref.object_key.startswith(expected_prefix):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"page image must be under {expected_prefix!r}; "
                    f"received object {ref.object_key!r}"
                ),
            )

        descriptor = PageDescriptor(
            page_id=item.page_id,
            page_number=item.page_number,
            filename=item.filename,
            metadata=item.metadata,
        )
        pages.append(
            await prepare_minio_page(
                image_url=item.image_url,
                document_id=payload.document_id,
                descriptor=descriptor,
                fallback_page_number=index,
                settings=settings,
                storage=storage,
            )
        )
    return pages


def _processing_error(run) -> str:
    failures: list[str] = []
    for page in run.response.pages:
        for module, status in page.modules.items():
            if status.state == ProcessingState.FAILED:
                failures.append(
                    f"page {page.page_number} {module.value}: {status.error or 'processing failed'}"
                )
    return "; ".join(failures) or f"document processing state={run.response.processing.state.value}"


async def _publish_debug_run(run) -> None:
    if run.response.processing.state != ProcessingState.SUCCESS:
        raise HTTPException(status_code=500, detail=_processing_error(run))
    try:
        await asyncio.to_thread(publisher.publish, run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MinioStorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/extract",
    response_model=DocumentExtractionResponse,
    summary="Run full extraction for uploaded local images",
    description=(
        "Local/development multipart workflow. Upload one or more page images and run OCR, "
        "figure/table, and stamp/signature pipelines through the same in-process orchestrator "
        "used by the product workflow. Set persist_outputs=true to also exercise the product "
        "artifact publisher while retaining the detailed inspection response."
    ),
)
async def extract_document(
    images: list[UploadFile] = File(..., description="1..N page images in document order."),
    document_id: str = Form(...),
    document_metadata_json: str | None = Form(None),
    pages_metadata_json: str | None = Form(None),
    persist_outputs: bool = Form(False),
):
    if not images:
        raise HTTPException(status_code=422, detail="at least one image is required")
    if len(images) > settings.max_pages_per_document:
        raise HTTPException(
            status_code=422,
            detail=f"document exceeds max_pages_per_document={settings.max_pages_per_document}",
        )

    request_id = new_request_id()
    document_metadata = parse_json_object(document_metadata_json, "document_metadata_json")
    descriptors = parse_page_descriptors(pages_metadata_json, len(images))

    pages = []
    for index, (upload, descriptor) in enumerate(zip(images, descriptors, strict=True), start=1):
        pages.append(
            await prepare_uploaded_page(
                upload=upload,
                document_id=document_id,
                descriptor=descriptor,
                fallback_page_number=index,
                settings=settings,
            )
        )

    if not persist_outputs:
        return await orchestrator.extract_document(
            document_id=document_id,
            pages=pages,
            request_id=request_id,
            document_metadata=document_metadata,
        )

    run = await orchestrator.extract_document_run(
        document_id=document_id,
        pages=pages,
        request_id=request_id,
        document_metadata=document_metadata,
    )
    await _publish_debug_run(run)
    return run.response


@router.post(
    "/extract/minio",
    response_model=ExtractionJobResponse,
    response_model_exclude_none=True,
    summary="Process one MinIO-backed document and persist AI artifacts",
    description=(
        "Primary Backend → AI production workflow. The backend sends the document identifier "
        "and all page image URLs. Wiki Hami validates and reads those images through MinIO, runs "
        "all three model pipelines, writes per-page module artifacts and visual crops back to the "
        "configured bucket under documents/<document_id>/, then returns only processing status. "
        "The request remains synchronous because the backend Celery task owns orchestration."
    ),
    responses={
        404: {"description": "A referenced MinIO object was not found"},
        422: {"description": "Invalid request, MinIO URL, image, or document identifier"},
        500: {"model": ExtractionJobResponse, "description": "One or more model pipelines failed"},
        502: {"model": ExtractionJobResponse, "description": "MinIO read/write/connectivity failure"},
        503: {"description": "MinIO integration disabled or misconfigured"},
    },
)
async def extract_minio_document(payload: MinioDocumentRequest):
    pages = await _prepare_minio_pages(payload)
    request_id = new_request_id()
    run = await orchestrator.extract_document_run(
        document_id=payload.document_id,
        pages=pages,
        request_id=request_id,
        document_metadata=payload.document_metadata,
    )

    if run.response.processing.state != ProcessingState.SUCCESS:
        return JSONResponse(
            status_code=500,
            content={
                "document_id": payload.document_id,
                "status": "failed",
                "error": _processing_error(run),
            },
        )

    try:
        await asyncio.to_thread(publisher.publish, run)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "document_id": payload.document_id,
                "status": "failed",
                "error": str(exc),
            },
        )
    except MinioStorageError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "document_id": payload.document_id,
                "status": "failed",
                "error": str(exc),
            },
        )

    return ExtractionJobResponse(document_id=payload.document_id, status="success")


@router.post(
    "/extract/minio/inspect",
    response_model=DocumentExtractionResponse,
    summary="Run full MinIO extraction and return detailed results for local inspection",
    description=(
        "Engineering/Streamlit companion to the product endpoint. It uses the same MinIO "
        "acquisition, preprocessing and model orchestration and returns the detailed canonical "
        "document response. Set persist_outputs=true to additionally publish the exact product "
        "artifacts without a second inference pass."
    ),
)
async def inspect_minio_document(
    payload: MinioDocumentRequest,
    persist_outputs: bool = Query(False),
):
    pages = await _prepare_minio_pages(payload)
    run = await orchestrator.extract_document_run(
        document_id=payload.document_id,
        pages=pages,
        request_id=new_request_id(),
        document_metadata=payload.document_metadata,
    )
    if persist_outputs:
        await _publish_debug_run(run)
    return run.response
