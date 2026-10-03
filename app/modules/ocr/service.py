from time import perf_counter

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import MockOCRBackend, create_ocr_backend
from app.preprocessing.transforms import restore_bbox_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.extraction import ModulePageResponse
from app.schemas.status import ModuleName, ModuleStatus, ProcessingState
from app.utils.ids import new_object_id


class OCRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._backend = create_ocr_backend(settings)

    def _mock_objects(self, page: PreparedPage) -> list[DetectedObject]:
        w, h = page.processed_image.size
        processed_bbox = BBox(x1=w * 0.08, y1=h * 0.08, x2=w * 0.92, y2=h * 0.25)
        source_bbox = restore_bbox_to_source(
            processed_bbox,
            page.transform,
            page.image_metadata.source_width,
            page.image_metadata.source_height,
        )
        return [
            DetectedObject(
                object_id=new_object_id("paragraph"),
                document_id=page.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                type=ObjectType.PARAGRAPH,
                bbox=source_bbox,
                confidence=0.99,
                text="MOCK_OCR_TEXT",
                raw_text="MOCK_OCR_TEXT",
                metadata={"mock": True},
                provenance=Provenance(
                    module=ModuleName.OCR,
                    backend="mock",
                    model_id=self.settings.ocr_model_id,
                ),
            )
        ]

    async def run(self, page: PreparedPage, request_id: str) -> ModulePageResponse:
        started = perf_counter()
        backend_metadata = self._backend.metadata

        if isinstance(self._backend, MockOCRBackend):
            objects = self._mock_objects(page)
        else:
            lines = await run_blocking(
                BlockingPool.OCR,
                self._backend.predict,
                page.processed_image,
            )
            objects = lines_to_detected_objects(
                lines,
                page=page,
                settings=self.settings,
                backend_metadata=backend_metadata,
            )

        duration = (perf_counter() - started) * 1000
        status = ModuleStatus(
            module=ModuleName.OCR,
            state=ProcessingState.SUCCESS,
            duration_ms=duration,
            model_id=backend_metadata.model_id,
            backend=backend_metadata.backend,
            warnings=[] if objects else ["no_text_detected"],
        )
        return ModulePageResponse(
            schema_version=self.settings.schema_version,
            request_id=request_id,
            document_id=page.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            module=ModuleName.OCR,
            image=page.image_metadata,
            transform=page.transform,
            objects=objects,
            status=status,
        )
