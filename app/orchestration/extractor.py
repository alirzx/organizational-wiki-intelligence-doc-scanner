import asyncio
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from app.core.config import Settings
from app.modules.figure_table.service import FigureTableService
from app.modules.ocr.service import OCRService
from app.modules.stamp_signature.service import StampSignatureService
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType
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
    ) -> PageRunResult:
        async with page_semaphore:
            started = perf_counter()
            jobs = [
                (ModuleName.OCR, self.ocr.run(page, request_id)),
                (ModuleName.FIGURE_TABLE, self.figure_table.run(page, request_id)),
                (ModuleName.STAMP_SIGNATURE, self.stamp_signature.run(page, request_id)),
            ]
            raw_results = await asyncio.gather(
                *(
                    asyncio.wait_for(job, timeout=self.settings.module_timeout_seconds)
                    for _, job in jobs
                ),
                return_exceptions=True,
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
                page_metadata=page.page_metadata,
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
    ) -> DocumentRunResult:
        started = perf_counter()
        # asyncio synchronization primitives are bound to the event loop that
        # first waits on them. Celery executes each task through asyncio.run(),
        # so a cached orchestrator may be reused across multiple event loops.
        # Keep the semaphore scoped to this document/run instead of the
        # process-local orchestrator instance.
        page_semaphore = asyncio.Semaphore(self.settings.page_concurrency)
        page_runs = await asyncio.gather(
            *(self._run_page(page, request_id, page_semaphore) for page in pages)
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
    ) -> DocumentExtractionResponse:
        """Backward-compatible merged extraction response used by local/dev callers."""
        run = await self.extract_document_run(
            document_id=document_id,
            pages=pages,
            request_id=request_id,
            document_metadata=document_metadata,
        )
        return run.response
