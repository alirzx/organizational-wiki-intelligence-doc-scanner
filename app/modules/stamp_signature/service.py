from time import perf_counter

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import Settings
from app.modules.stamp_signature.adapter import mark_items_to_detected_objects
from app.modules.stamp_signature.rfdetr_backend import RFDETRStampSignatureBackend
from app.preprocessing.transforms import restore_bbox_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.extraction import ModulePageResponse
from app.schemas.status import ModuleName, ModuleStatus, ProcessingState
from app.utils.ids import new_object_id


class StampSignatureService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._rfdetr_backend = (
            RFDETRStampSignatureBackend(settings)
            if settings.stamp_signature_backend.lower() == "rfdetr"
            else None
        )

    def _mock_objects(self, page: PreparedPage) -> list[DetectedObject]:
        w, h = page.processed_image.size
        specs = [
            (ObjectType.STAMP, BBox(x1=w * 0.62, y1=h * 0.72, x2=w * 0.88, y2=h * 0.90)),
            (ObjectType.SIGNATURE, BBox(x1=w * 0.52, y1=h * 0.82, x2=w * 0.76, y2=h * 0.94)),
        ]
        objects: list[DetectedObject] = []
        for obj_type, processed_bbox in specs:
            source_bbox = restore_bbox_to_source(
                processed_bbox,
                page.transform,
                page.image_metadata.source_width,
                page.image_metadata.source_height,
            )
            objects.append(
                DetectedObject(
                    object_id=new_object_id(obj_type.value),
                    document_id=page.document_id,
                    page_id=page.page_id,
                    page_number=page.page_number,
                    type=obj_type,
                    bbox=source_bbox,
                    confidence=0.97,
                    metadata={"mock": True, "source_classes_filtered": ["stamp", "signature"]},
                    provenance=Provenance(
                        module=ModuleName.STAMP_SIGNATURE,
                        backend="mock",
                        model_id=self.settings.stamp_signature_model_id,
                    ),
                )
            )
        return objects

    async def run(self, page: PreparedPage, request_id: str) -> ModulePageResponse:
        started = perf_counter()
        backend = self.settings.stamp_signature_backend.lower()
        if backend == "mock":
            objects = self._mock_objects(page)
        elif backend == "rfdetr":
            assert self._rfdetr_backend is not None
            items = await run_blocking(
                BlockingPool.STAMP_SIGNATURE,
                self._rfdetr_backend.predict,
                page.processed_image,
            )
            objects = mark_items_to_detected_objects(
                items,
                page=page,
                settings=self.settings,
                backend_name="rfdetr",
            )
        else:
            raise RuntimeError(
                f"Unsupported stamp/signature backend: {self.settings.stamp_signature_backend}"
            )

        duration = (perf_counter() - started) * 1000
        return ModulePageResponse(
            schema_version=self.settings.schema_version,
            request_id=request_id,
            document_id=page.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            module=ModuleName.STAMP_SIGNATURE,
            image=page.image_metadata,
            transform=page.transform,
            objects=objects,
            status=ModuleStatus(
                module=ModuleName.STAMP_SIGNATURE,
                state=ProcessingState.SUCCESS,
                duration_ms=duration,
                model_id=self.settings.stamp_signature_model_id,
                backend=backend,
                warnings=[] if objects else ["no_stamp_or_signature_detected"],
            ),
        )
