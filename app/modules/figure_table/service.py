from time import perf_counter

from app.core.blocking import BlockingPool, run_blocking
from app.core.config import Settings
from app.modules.figure_table.adapter import layout_items_to_detected_objects
from app.modules.figure_table.pp_doclayout_backend import PPDocLayoutBackend
from app.preprocessing.transforms import restore_bbox_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.extraction import ModulePageResponse
from app.schemas.status import ModuleName, ModuleStatus, ProcessingState
from app.utils.ids import new_object_id


class FigureTableService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pp_backend = (
            PPDocLayoutBackend(settings)
            if settings.figure_table_backend.lower() == "pp_doclayout"
            else None
        )

    def _mock_objects(self, page: PreparedPage) -> list[DetectedObject]:
        w, h = page.processed_image.size
        specs = [
            (ObjectType.TABLE, BBox(x1=w * 0.08, y1=h * 0.35, x2=w * 0.92, y2=h * 0.57)),
            (ObjectType.FIGURE, BBox(x1=w * 0.12, y1=h * 0.63, x2=w * 0.55, y2=h * 0.85)),
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
                    confidence=0.98,
                    metadata={"mock": True},
                    provenance=Provenance(
                        module=ModuleName.FIGURE_TABLE,
                        backend="mock",
                        model_id=self.settings.figure_table_model_id,
                    ),
                )
            )
        return objects

    async def run(self, page: PreparedPage, request_id: str) -> ModulePageResponse:
        started = perf_counter()
        backend = self.settings.figure_table_backend.lower()
        if backend == "mock":
            objects = self._mock_objects(page)
        elif backend == "pp_doclayout":
            assert self._pp_backend is not None
            items = await run_blocking(
                BlockingPool.FIGURE_TABLE,
                self._pp_backend.predict,
                page.processed_image,
            )
            objects = layout_items_to_detected_objects(
                items,
                page=page,
                settings=self.settings,
                backend_name="pp_doclayout",
            )
        else:
            raise RuntimeError(f"Unsupported figure/table backend: {self.settings.figure_table_backend}")

        duration = (perf_counter() - started) * 1000
        return ModulePageResponse(
            schema_version=self.settings.schema_version,
            request_id=request_id,
            document_id=page.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            module=ModuleName.FIGURE_TABLE,
            image=page.image_metadata,
            transform=page.transform,
            objects=objects,
            status=ModuleStatus(
                module=ModuleName.FIGURE_TABLE,
                state=ProcessingState.SUCCESS,
                duration_ms=duration,
                model_id=self.settings.figure_table_model_id,
                backend=backend,
                warnings=[] if objects else ["no_figure_or_table_detected"],
            ),
        )
