import asyncio
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from app.core.config import Settings
from app.modules.figure_table.service import FigureTableService
from app.modules.ocr.service import OCRService
from app.modules.ocr.adapter import groups_to_detected_objects
from app.modules.stamp_signature.service import StampSignatureService
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType
from app.schemas.extraction import ContentGroup as ContentGroupSchema, DocumentExtractionResponse, GroupingDiagnostics, ModulePageResponse, PageExtractionResponse
from app.schemas.status import ModuleName, ModuleStatus, ProcessingState, ProcessingStatus
from app.text_processing.candidate_generator import CandidateConfig
from app.text_processing.classifiers.base import BlockRelationshipClassifier, GroupingModelError
from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
from app.text_processing.feature_extractor import feature_schema
from app.text_processing.types import GroupingMode, TextBlock
from app.text_processing.embeddings.base import TextEmbedder
from app.orchestration.paragraph_resolver import resolve_paragraphs


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
        grouping_classifier: BlockRelationshipClassifier | None = None,
        grouping_embedder: TextEmbedder | None = None,
    ):
        self.settings = settings
        self.ocr = ocr or OCRService(settings)
        self.figure_table = figure_table or FigureTableService(settings)
        self.stamp_signature = stamp_signature or StampSignatureService(settings)
        self.grouping_classifier = grouping_classifier
        self.grouping_embedder = grouping_embedder

    def _apply_learned_grouping(self, document_id: str, page_runs: list[PageRunResult]):
        if not self.settings.grouping_enabled:
            return [], GroupingDiagnostics(mode=GroupingMode.HEURISTIC_DISABLED)
        classifier = self.grouping_classifier or LightGBMRelationshipClassifier(
            self.settings.grouping_model_path, feature_schema()
        )
        blocks: list[TextBlock] = []
        for page_run in page_runs:
            ocr_result = page_run.modules.get(ModuleName.OCR)
            if ocr_result is None:
                continue
            ordinal = 0
            for obj in ocr_result.objects:
                if obj.type != ObjectType.PARAGRAPH:
                    continue
                blocks.append(TextBlock(
                    block_id=obj.object_id, document_id=document_id, page_id=obj.page_id,
                    page_number=obj.page_number, content_ordinal=ordinal,
                    original_text=obj.raw_text or obj.text or "", normalized_text=obj.text or obj.raw_text or "",
                    bbox=obj.bbox, polygon=obj.polygon, ocr_confidence=obj.confidence,
                    page_width=page_run.response.image.source_width,
                    page_height=page_run.response.image.source_height,
                    metadata={"source_object_id": obj.object_id},
                ))
                ordinal += 1
        if not blocks:
            return [], GroupingDiagnostics(mode=GroupingMode.LEARNED, model_package_id=classifier.package_id)
        resolution = resolve_paragraphs(
            blocks, classifier,
            candidate_config=CandidateConfig(
                reading_lookahead=self.settings.grouping_candidate_reading_window,
                max_pairs=self.settings.grouping_candidate_max_pairs_per_block * max(1, len(blocks)),
                cross_page_window=self.settings.grouping_candidate_cross_page_window,
                max_page_distance=self.settings.grouping_candidate_max_page_distance,
            ),
            embedder=self.grouping_embedder,
            semantic_failure_policy=self.settings.semantic_failure_policy,
        )
        projections = groups_to_detected_objects(
            resolution.groups, document_id=document_id, backend_name="grouping",
            model_id=classifier.package_id,
        )
        for page_run in page_runs:
            local = [item for item in projections if item.page_number == page_run.response.page_number]
            ocr_result = page_run.modules.get(ModuleName.OCR)
            if ocr_result is not None:
                page_run.modules[ModuleName.OCR] = ocr_result.model_copy(update={"objects": local})
            other = [item for item in page_run.response.objects if item.type != ObjectType.PARAGRAPH]
            combined = sorted(other + local, key=lambda item: (item.bbox.y1, item.bbox.x1, item.type.value))
            page_run.response = page_run.response.model_copy(update={"objects": combined})
        public_groups = [ContentGroupSchema.model_validate({
            "group_id": group.group_id, "group_order": group.group_order,
            "group_type": group.group_type, "text": group.text, "raw_text": group.raw_text,
            "confidence": group.confidence, "confidence_method": group.confidence_method,
            "members": [member.__dict__ for member in group.members],
            "page_spans": [span.__dict__ for span in group.page_spans], "cross_page": group.cross_page,
            "metadata": group.metadata,
        }) for group in resolution.groups]
        diagnostics = GroupingDiagnostics(
            mode=(GroupingMode.LEARNED if resolution.semantic_mode != "unavailable" else GroupingMode.LEARNED_WITHOUT_SEMANTICS),
            model_package_id=classifier.package_id,
            feature_schema_version=feature_schema().version,
            semantic_mode=resolution.semantic_mode,
            counts={"blocks": len(blocks), "candidates": len(resolution.predictions), "groups": len(public_groups)},
            timings_ms=resolution.metrics or {},
        )
        return public_groups, diagnostics

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
        try:
            content_groups, grouping = self._apply_learned_grouping(document_id, page_runs)
        except (GroupingModelError, ValueError, RuntimeError) as exc:
            if self.settings.semantic_failure_policy == "fail_fast":
                raise
            content_groups = []
            grouping = GroupingDiagnostics(
                mode=GroupingMode.HEURISTIC_FALLBACK,
                fallback_reason=type(exc).__name__,
                semantic_mode="unavailable" if self.settings.semantic_features_enabled else "disabled",
            )
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
            content_groups=content_groups,
            grouping=grouping,
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
