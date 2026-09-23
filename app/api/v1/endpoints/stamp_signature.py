from fastapi import APIRouter

from app.api.v1.request_parsing import prepare_minio_page
from app.core.config import get_settings
from app.core.runtime import get_minio_storage_service, get_stamp_signature_service
from app.schemas.extraction import ModulePageResponse
from app.schemas.image import PageDescriptor
from app.schemas.storage import ModuleImageRequest
from app.utils.ids import default_page_id, new_request_id

router = APIRouter(tags=["Stamp / Signature"])
settings = get_settings()
service = get_stamp_signature_service()
storage = get_minio_storage_service()


@router.post(
    "/stamp-signature",
    response_model=ModulePageResponse,
    summary="Detect stamps and signatures for one MinIO image",
    description=(
        "Engineering/debug mark-detection endpoint. Send one MinIO image URL to exercise the same "
        "RF-DETR service used by the product document workflow. Wiki Hami validates and reads the "
        "object through MinIO, applies shared preprocessing and returns canonical stamp/signature "
        "objects. Product backend integration uses /extract/minio."
    ),
    responses={
        404: {"description": "MinIO object not found"},
        422: {"description": "Invalid MinIO URL, request, or image"},
        502: {"description": "MinIO read/connectivity failure"},
        503: {"description": "MinIO integration disabled or misconfigured"},
    },
)
async def run_stamp_signature(payload: ModuleImageRequest):
    request_id = new_request_id()
    descriptor = PageDescriptor(
        page_id=payload.page_id or default_page_id(payload.document_id, payload.page_number),
        page_number=payload.page_number,
        metadata=payload.page_metadata,
    )
    page = await prepare_minio_page(
        image_url=payload.image_url,
        document_id=payload.document_id,
        descriptor=descriptor,
        fallback_page_number=payload.page_number,
        settings=settings,
        storage=storage,
    )
    return await service.run(page, request_id)
