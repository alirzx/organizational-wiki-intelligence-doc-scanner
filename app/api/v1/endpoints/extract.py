import asyncio
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

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
    get_job_store,
    get_minio_storage_service,
)
from app.jobs.tasks import process_minio_document
from app.schemas.extraction import DocumentExtractionResponse
from app.schemas.image import PageDescriptor
from app.schemas.status import ProcessingState
from app.schemas.storage import ExtractionJobResponse, MinioDocumentRequest
from app.security import verify_backend_api_key
from app.storage.minio_service import MinioConfigurationError, MinioStorageError, MinioUrlError
from app.utils.ids import new_request_id

router = APIRouter(tags=["Full Extraction"])
settings = get_settings()
orchestrator = get_extraction_orchestrator()
storage = get_minio_storage_service()
publisher = get_artifact_publisher()


def _validate_minio_request(payload: MinioDocumentRequest) -> None:
    if len(payload.pages) > settings.max_pages_per_document:
        raise HTTPException(status_code=422, detail=f"document exceeds max_pages_per_document={settings.max_pages_per_document}")
    expected_prefix = f"documents/{payload.document_id}/images/"
    for item in payload.pages:
        try:
            ref = storage.parse_image_url(item.image_url)
        except MinioUrlError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MinioConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not ref.object_key.startswith(expected_prefix):
            raise HTTPException(
                status_code=422,
                detail=f"page image must be under {expected_prefix!r}; received object {ref.object_key!r}",
            )


async def _prepare_minio_pages(payload: MinioDocumentRequest):
    _validate_minio_request(payload)
    pages = []
    for index, item in enumerate(payload.pages, start=1):
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
        for module, module_status in page.modules.items():
            if module_status.state == ProcessingState.FAILED:
                failures.append(f"page {page.page_number} {module.value}: {module_status.error or 'processing failed'}")
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


@router.post("/extract", response_model=DocumentExtractionResponse, summary="Run full extraction for uploaded local images")
async def extract_document(
    images: list[UploadFile] = File(...),
    document_id: str = Form(...),
    document_metadata_json: str | None = Form(None),
    pages_metadata_json: str | None = Form(None),
    persist_outputs: bool = Form(False),
):
    if not images:
        raise HTTPException(status_code=422, detail="at least one image is required")
    if len(images) > settings.max_pages_per_document:
        raise HTTPException(status_code=422, detail=f"document exceeds max_pages_per_document={settings.max_pages_per_document}")
    request_id = new_request_id()
    document_metadata = parse_json_object(document_metadata_json, "document_metadata_json")
    descriptors = parse_page_descriptors(pages_metadata_json, len(images))
    pages = []
    for index, (upload, descriptor) in enumerate(zip(images, descriptors, strict=True), start=1):
        pages.append(await prepare_uploaded_page(
            upload=upload,
            document_id=document_id,
            descriptor=descriptor,
            fallback_page_number=index,
            settings=settings,
        ))
    if not persist_outputs:
        return await orchestrator.extract_document(
            document_id=document_id, pages=pages, request_id=request_id, document_metadata=document_metadata
        )
    run = await orchestrator.extract_document_run(
        document_id=document_id, pages=pages, request_id=request_id, document_metadata=document_metadata
    )
    await _publish_debug_run(run)
    return run.response


@router.post(
    "/extract/minio",
    response_model=ExtractionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_backend_api_key)],
    summary="Accept one MinIO-backed document for asynchronous extraction",
)
async def extract_minio_document(payload: MinioDocumentRequest):
    _validate_minio_request(payload)
    job_id = uuid4().hex
    job_store = get_job_store()
    try:
        job_store.create(job_id, payload.document_id)
        process_minio_document.apply_async(
            args=[payload.model_dump(mode="json"), job_id],
            task_id=job_id,
            queue=settings.celery_queue,
        )
    except Exception as exc:
        try:
            job_store.update(job_id, status="failed", error={"code": "QUEUE_UNAVAILABLE", "message": str(exc)})
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=f"could not enqueue extraction job: {exc}") from exc
    return ExtractionJobResponse(job_id=job_id, document_id=payload.document_id, status="queued")


@router.post(
    "/extract/minio/inspect",
    response_model=DocumentExtractionResponse,
    summary="Run full MinIO extraction synchronously for engineering inspection",
)
async def inspect_minio_document(payload: MinioDocumentRequest, persist_outputs: bool = Query(False)):
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
