import asyncio
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from time import perf_counter

from app.core.config import Settings
from app.modules.figure_table.service import FigureTableService
from app.modules.ocr.service import OCRService, UnreadableTextRegionsError
from app.modules.ocr.native_text import covered_area
from app.schemas.storage import NativeTextPage
from app.utils.ids import new_object_id
from app.preprocessing.transforms import restore_bbox_to_source
from app.modules.stamp_signature.service import StampSignatureService
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.extraction import DocumentExtractionResponse, ModulePageResponse, PageExtractionResponse
from app.schemas.status import ModuleName, ModuleStatus, ProcessingState, ProcessingStatus


@dataclass
class PageRunResult:
    page: PreparedPage
    modules: dict[ModuleName, ModulePageResponse | None]
    response: PageExtractionResponse


@dataclass
class DocumentRunResult:
    response: DocumentExtractionResponse
    pages: list[PageRunResult]


class ExtractionOrchestrator:
    _MODULE_ORDER = (
        ModuleName.OCR,
        ModuleName.FIGURE_TABLE,
        ModuleName.STAMP_SIGNATURE,
    )

    def __init__(
        self,
        settings: Settings,
        *,
        ocr: OCRService | None = None,
        figure_table: FigureTableService | None = None,
        stamp_signature: StampSignatureService | None = None,
    ):
        self.settings = settings
        self.ocr = ocr or OCRService(settings)
        self.figure_table = figure_table or FigureTableService(settings)
        self.stamp_signature = stamp_signature or StampSignatureService(settings)

    @staticmethod
    def _page_state(statuses: list[ModuleStatus]) -> ProcessingState:
        successes = sum(s.state == ProcessingState.SUCCESS for s in statuses)
        if successes == len(statuses):
            return ProcessingState.SUCCESS
        if successes == 0:
            return ProcessingState.FAILED
        return ProcessingState.PARTIAL_SUCCESS

    def _module_timeout(self, module: ModuleName) -> float:
        if module == ModuleName.OCR and self.settings.text_extraction_mode == "vlm":
            return self.settings.vlm_module_timeout_seconds
        return self.settings.module_timeout_seconds

    def _failure_status(self, module: ModuleName, exc: BaseException) -> ModuleStatus:
        return ModuleStatus(
            module=module,
            state=ProcessingState.FAILED,
            duration_ms=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    async def _run_page(
        self,
        page: PreparedPage,
        request_id: str,
        page_semaphore: asyncio.Semaphore,
        selected_modules: tuple[ModuleName, ...],
    ) -> PageRunResult:
        async with page_semaphore:
            started = perf_counter()
            services = {
                ModuleName.OCR: self.ocr,
                ModuleName.FIGURE_TABLE: self.figure_table,
                ModuleName.STAMP_SIGNATURE: self.stamp_signature,
            }
            async def run_module(module):
                if module == ModuleName.OCR and "_pdf_text" in page.page_metadata:
                    figures = []
                    if layout_task is not None:
                        layout_result = await layout_task
                        if isinstance(layout_result, ModulePageResponse):
                            figures = layout_result.objects
                    return await self.ocr.run_with_native_text(page, request_id, figures)
                if module == ModuleName.FIGURE_TABLE and layout_task is not None:
                    return await layout_task
                return await services[module].run(page, request_id)

            async def run_layout():
                try:
                    return await asyncio.wait_for(
                        self.figure_table.run(page, request_id),
                        timeout=self._module_timeout(ModuleName.FIGURE_TABLE),
                    )
                except Exception as exc:
                    return exc

            layout_task = (
                asyncio.create_task(run_layout())
                if "_pdf_text" in page.page_metadata and ModuleName.FIGURE_TABLE in selected_modules
                else None
            )
            jobs = [(module, run_module(module)) for module in selected_modules]
            raw_results = await asyncio.gather(
                *(
                    asyncio.wait_for(job, timeout=self._module_timeout(module))
                    for module, job in jobs
                ),
                return_exceptions=True,
            )

            if "_pdf_text" in page.page_metadata and ModuleName.FIGURE_TABLE in selected_modules:
                native = NativeTextPage.model_validate(page.page_metadata["_pdf_text"])
                result = raw_results[selected_modules.index(ModuleName.FIGURE_TABLE)]
                if isinstance(result, ModulePageResponse):
                    for box in native.image_regions:
                        existing = [obj.bbox for obj in result.objects if obj.type == ObjectType.FIGURE]
                        if covered_area(box, existing) >= 0.85:
                            continue
                        result.objects.append(DetectedObject(
                            object_id=new_object_id("figure"), document_id=page.document_id,
                            page_id=page.page_id, page_number=page.page_number,
                            type=ObjectType.FIGURE, bbox=box, confidence=1.0,
                            metadata={"source_label": "pdf_embedded_image"},
                            provenance=Provenance(module=ModuleName.FIGURE_TABLE, backend="pdf_source", model_id="pdf-image-layer"),
                        ))
                    if result.objects:
                        result.status.warnings = [w for w in result.status.warnings if w != "no_figure_or_table_detected"]

            # Calligraphy/logos can trigger text detection. Require independent
            # figure localization over every unreadable region before accepting
            # an image-only result; unreadable body text must still fail.
            results_by_module = dict(zip(selected_modules, raw_results, strict=True))
            ocr_error = results_by_module.get(ModuleName.OCR)
            layout_result = results_by_module.get(ModuleName.FIGURE_TABLE)
            if (
                isinstance(ocr_error, UnreadableTextRegionsError)
                and ocr_error.boxes
                and isinstance(layout_result, ModulePageResponse)
                and layout_result.status.state == ProcessingState.SUCCESS
            ):
                figures = [
                    obj.bbox for obj in layout_result.objects
                    if obj.type == ObjectType.FIGURE and obj.confidence >= 0.8
                ]
                boxes = [
                    restore_bbox_to_source(
                        box, page.transform,
                        page.image_metadata.source_width, page.image_metadata.source_height,
                    )
                    for box in ocr_error.boxes
                ]

                def covered_by_figure(box):
                    area = (box.x2 - box.x1) * (box.y2 - box.y1)
                    return area > 0 and any(
                        max(0, min(box.x2, figure.x2) - max(box.x1, figure.x1))
                        * max(0, min(box.y2, figure.y2) - max(box.y1, figure.y1))
                        / area >= 0.75
                        for figure in figures
                    )

                if all(covered_by_figure(box) for box in boxes):
                    raw_results[selected_modules.index(ModuleName.OCR)] = ModulePageResponse(
                        module=ModuleName.OCR,
                        schema_version=self.settings.schema_version,
                        request_id=request_id,
                        document_id=page.document_id,
                        page_id=page.page_id,
                        page_number=page.page_number,
                        image=page.image_metadata,
                        transform=page.transform,
                        objects=[],
                        status=ModuleStatus(
                            module=ModuleName.OCR,
                            state=ProcessingState.SUCCESS,
                            duration_ms=(perf_counter() - started) * 1000,
                            backend="paddle",
                            model_id=self.settings.ocr_model_id,
                            warnings=["unreadable_figure_text_preserved_as_image"],
                        ),
                    )

            statuses: list[ModuleStatus] = []
            objects: list[DetectedObject] = []
            warnings: list[str] = []
            module_map: dict[ModuleName, ModuleStatus] = {}
            module_results: dict[ModuleName, ModulePageResponse | None] = {}

            for (module_name, _), result in zip(jobs, raw_results, strict=True):
                if isinstance(result, BaseException):
                    status = self._failure_status(module_name, result)
                    module_results[module_name] = None
                    warnings.append(f"{module_name.value}_failed")
                else:
                    assert isinstance(result, ModulePageResponse)
                    status = result.status
                    module_results[module_name] = result
                    objects.extend(result.objects)
                    warnings.extend(result.status.warnings)
                statuses.append(status)
                module_map[module_name] = status

            objects.sort(key=lambda x: (x.bbox.y1, x.bbox.x1, x.type.value))
            duration = (perf_counter() - started) * 1000
            response = PageExtractionResponse(
                schema_version=self.settings.schema_version,
                request_id=request_id,
                document_id=page.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                page_metadata={key: value for key, value in page.page_metadata.items() if key != "_pdf_text"},
                image=page.image_metadata,
                transform=page.transform,
                objects=objects,
                modules=module_map,
                processing=ProcessingStatus(
                    state=self._page_state(statuses),
                    duration_ms=duration,
                    warnings=warnings,
                ),
            )
            return PageRunResult(page=page, modules=module_results, response=response)

    async def extract_document_run(
        self,
        *,
        document_id: str,
        pages: list[PreparedPage],
        request_id: str,
        document_metadata: dict,
        selected_modules: Collection[ModuleName] | None = None,
    ) -> DocumentRunResult:
        started = perf_counter()
        selected = set(selected_modules) if selected_modules is not None else set(self._MODULE_ORDER)
        ordered_modules = tuple(module for module in self._MODULE_ORDER if module in selected)
        if not ordered_modules:
            raise ValueError("at least one extraction module must be selected")
        # asyncio synchronization primitives are bound to the event loop that
        # first waits on them. Celery executes each task through asyncio.run(),
        # so a cached orchestrator may be reused across multiple event loops.
        # Keep the semaphore scoped to this document/run instead of the
        # process-local orchestrator instance.
        page_semaphore = asyncio.Semaphore(
            1 if self.settings.text_extraction_mode == "vlm" else self.settings.page_concurrency
        )
        page_runs = await asyncio.gather(
            *(
                self._run_page(page, request_id, page_semaphore, ordered_modules)
                for page in pages
            )
        )
        page_runs = sorted(page_runs, key=lambda item: item.response.page_number)
        page_results = [item.response for item in page_runs]

        objects = [obj for page in page_results for obj in page.objects]
        objects.sort(key=lambda x: (x.page_number, x.bbox.y1, x.bbox.x1, x.type.value))

        counts = Counter(obj.type for obj in objects)
        object_counts = {obj_type: counts.get(obj_type, 0) for obj_type in ObjectType}

        states = [page.processing.state for page in page_results]
        if states and all(state == ProcessingState.SUCCESS for state in states):
            state = ProcessingState.SUCCESS
        elif states and all(state == ProcessingState.FAILED for state in states):
            state = ProcessingState.FAILED
        else:
            state = ProcessingState.PARTIAL_SUCCESS

        warnings = [
            warning
            for page in page_results
            for warning in page.processing.warnings
        ]
        duration = (perf_counter() - started) * 1000

        response = DocumentExtractionResponse(
            schema_version=self.settings.schema_version,
            request_id=request_id,
            document_id=document_id,
            document_metadata=document_metadata,
            page_count=len(page_results),
            pages=page_results,
            objects=objects,
            object_counts=object_counts,
            processing=ProcessingStatus(
                state=state,
                duration_ms=duration,
                warnings=warnings,
            ),
        )
        return DocumentRunResult(response=response, pages=page_runs)

    async def extract_document(
        self,
        *,
        document_id: str,
        pages: list[PreparedPage],
        request_id: str,
        document_metadata: dict,
        selected_modules: Collection[ModuleName] | None = None,
    ) -> DocumentExtractionResponse:
        """Backward-compatible merged extraction response used by local/dev callers."""
        run = await self.extract_document_run(
            document_id=document_id,
            pages=pages,
            request_id=request_id,
            document_metadata=document_metadata,
            selected_modules=selected_modules,
        )
        return run.response
