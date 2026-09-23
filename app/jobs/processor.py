from __future__ import annotations

import asyncio

from app.api.v1.request_parsing import prepare_minio_page
from app.core.config import get_settings
from app.core.runtime import get_artifact_publisher, get_extraction_orchestrator, get_minio_storage_service
from app.schemas.image import PageDescriptor
from app.schemas.status import ProcessingState
from app.schemas.storage import MinioDocumentRequest
from app.utils.ids import new_request_id


def _processing_error(run) -> str:
    failures: list[str] = []
    for page in run.response.pages:
        for module, status in page.modules.items():
            if status.state == ProcessingState.FAILED:
                failures.append(
                    f"page {page.page_number} {module.value}: {status.error or 'processing failed'}"
                )
    return "; ".join(failures) or f"document processing state={run.response.processing.state.value}"


async def process_minio_payload(payload: MinioDocumentRequest) -> list[str]:
    settings = get_settings()
    storage = get_minio_storage_service()
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

    run = await get_extraction_orchestrator().extract_document_run(
        document_id=payload.document_id,
        pages=pages,
        request_id=new_request_id(),
        document_metadata=payload.document_metadata,
    )
    if run.response.processing.state != ProcessingState.SUCCESS:
        raise RuntimeError(_processing_error(run))
    return await asyncio.to_thread(get_artifact_publisher().publish, run)
