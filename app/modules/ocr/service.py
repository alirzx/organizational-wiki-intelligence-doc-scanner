from dataclasses import replace
from time import perf_counter

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import MockOCRBackend, create_ocr_backend
from app.modules.ocr.deepseek_ocr_vlm_backend import DeepSeekOCRVLMBackend, VLMQualityError
from app.modules.ocr.paddle_backend import PaddleOCRBackend
from app.modules.ocr.text_normalization import normalize_persian_ocr_text
from app.modules.ocr.vlm_postprocessing import clean_vlm_lines
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
        self._fallback = None

    def _predict(self, image):
        primary = self._backend.metadata
        if not isinstance(self._backend, DeepSeekOCRVLMBackend):
            return self._backend.predict(image), primary, []

        warnings: list[str] = []

        # Persian/legal documents are read top-to-bottom. Use three horizontal
        # regions as the primary VLM strategy so dense pages cannot silently lose
        # their tail at the model generation limit and so each accepted block has
        # meaningful page geometry for layout.json.
        if self.settings.vlm_region_fallback:
            try:
                lines = clean_vlm_lines(self._backend.predict_regions(image))
                metadata = replace(primary, object_metadata={
                    **primary.object_metadata,
                    "ocr_strategy": "top_to_bottom_regions",
                })
                return lines, metadata, warnings
            except VLMQualityError as exc:
                warnings.append(f"vlm_regions_rejected: {exc}")

        # Keep the bounded whole-page path as a secondary recovery strategy. It
        # remains useful for sparse pages and preserves the previous behavior when
        # region OCR is explicitly disabled.
        try:
            lines = clean_vlm_lines(self._backend.predict(image))
            metadata = replace(primary, object_metadata={
                **primary.object_metadata,
                "ocr_strategy": "full_page_recovery",
            })
            return lines, metadata, warnings + ["ocr_recovered_by_full_page_vlm"]
        except VLMQualityError as exc:
            warnings.append(f"vlm_page_rejected: {exc}")

        if not self.settings.vlm_classic_fallback:
            raise VLMQualityError("Top-to-bottom region OCR and whole-page VLM OCR failed")

        if self._fallback is None:
            self._fallback = PaddleOCRBackend(self.settings)
        lines = self._fallback.predict(image)
        if not lines and self._fallback.last_detection_count:
            raise VLMQualityError("Text regions were detected but neither OCR backend could read them")
        lines = [
            replace(
                line,
                text=normalize_persian_ocr_text(line.text),
                raw_text=line.raw_text if line.raw_text is not None else line.text,
            )
            for line in lines
        ]
        metadata = replace(self._fallback.metadata, object_metadata={
            "text_extraction_mode": "ocr",
            "ocr_strategy": "paddle_fallback",
            "primary_model_id": primary.model_id,
        })
        return lines, metadata, warnings + ["ocr_recovered_by_paddle"]

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
        warnings = []

        if isinstance(self._backend, MockOCRBackend):
            objects = self._mock_objects(page)
        else:
            lines, backend_metadata, warnings = await run_blocking(
                BlockingPool.OCR,
                self._predict,
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
            warnings=warnings + ([] if objects else ["no_text_detected"]),
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
