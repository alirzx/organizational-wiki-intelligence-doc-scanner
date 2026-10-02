import asyncio
from time import perf_counter

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects, lines_to_text_blocks, objects_to_text_blocks
from app.modules.ocr.paddle_backend import PaddleOCRBackend
from app.modules.ocr.types import OCRLine, OCRAnalysis
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
        self._paddle_backend = (
            PaddleOCRBackend(settings) if settings.ocr_backend.lower() == "paddle" else None
        )

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
        response, _ = await self._run(page,request_id,retain_analysis=False)
        return response

    async def run_with_analysis(self, page: PreparedPage, request_id: str) -> tuple[ModulePageResponse, OCRAnalysis | None]:
        return await self._run(page,request_id,retain_analysis=True)

    async def _run(self, page: PreparedPage, request_id: str, *, retain_analysis: bool) -> tuple[ModulePageResponse, OCRAnalysis | None]:
        started = perf_counter()
        backend = self.settings.ocr_backend.lower()
        lines = []

        if backend == "mock":
            objects = self._mock_objects(page)
        elif backend == "paddle":
            assert self._paddle_backend is not None
            lines: list[OCRLine] = await asyncio.to_thread(self._paddle_backend.predict, page.processed_image)
            objects = lines_to_detected_objects(
                lines,
                page=page,
                settings=self.settings,
                backend_name="paddle",
            )
        else:
            raise RuntimeError(f"Unsupported OCR backend: {self.settings.ocr_backend}")

        duration = (perf_counter() - started) * 1000
        status = ModuleStatus(
            module=ModuleName.OCR,
            state=ProcessingState.SUCCESS,
            duration_ms=duration,
            model_id=self.settings.ocr_model_id,
            backend=backend,
            warnings=[] if objects else ["no_text_detected"],
        )
        response = ModulePageResponse(
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
        analysis = None
        if retain_analysis:
            blocks = (lines_to_text_blocks(lines, page=page) if backend == 'paddle'
                      else objects_to_text_blocks(objects, page=page))
            analysis = OCRAnalysis(tuple(lines), tuple(blocks), tuple(objects))
        return response, analysis
