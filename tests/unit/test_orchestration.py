from io import BytesIO
import asyncio

import pytest
from PIL import Image

from app.core.config import Settings
from app.orchestration.extractor import ExtractionOrchestrator
from app.preprocessing.pipeline import prepare_page


class FailingStampService:
    async def run(self, page, request_id):
        raise RuntimeError("test backend unavailable")


def _page(settings: Settings, *, page_number: int = 1):
    image = Image.new("RGB", (300, 400), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return prepare_page(
        data=buffer.getvalue(),
        filename=f"page-{page_number}.png",
        mime_type="image/png",
        document_id="doc_partial",
        page_id=f"doc_partial:p{page_number}",
        page_number=page_number,
        page_metadata={"source": "test"},
        settings=settings,
    )


@pytest.mark.asyncio
async def test_module_failure_keeps_successful_objects_and_reports_partial_success():
    settings = Settings(
        ocr_backend="mock",
        figure_table_backend="mock",
        stamp_signature_backend="mock",
    )
    orchestrator = ExtractionOrchestrator(settings, stamp_signature=FailingStampService())
    response = await orchestrator.extract_document(
        document_id="doc_partial",
        pages=[_page(settings)],
        request_id="req_test",
        document_metadata={},
    )

    assert response.processing.state == "partial_success"
    assert response.pages[0].processing.state == "partial_success"
    assert response.pages[0].modules["stamp_signature"].state == "failed"
    assert "test backend unavailable" in response.pages[0].modules["stamp_signature"].error
    assert {obj.type for obj in response.objects} == {"paragraph", "table", "figure"}
    assert all(
        obj.document_id == "doc_partial" and obj.page_id == "doc_partial:p1"
        for obj in response.objects
    )


def test_cached_orchestrator_can_be_reused_across_fresh_event_loops():
    settings = Settings(
        ocr_backend="mock",
        figure_table_backend="mock",
        stamp_signature_backend="mock",
        page_concurrency=1,
    )
    orchestrator = ExtractionOrchestrator(settings)
    pages = [_page(settings, page_number=1), _page(settings, page_number=2)]

    async def run_once(request_id: str):
        return await orchestrator.extract_document(
            document_id="doc_partial",
            pages=pages,
            request_id=request_id,
            document_metadata={},
        )

    first = asyncio.run(run_once("req_loop_1"))
    second = asyncio.run(run_once("req_loop_2"))

    assert first.processing.state == "success"
    assert second.processing.state == "success"
    assert [page.page_number for page in second.pages] == [1, 2]


def test_module_endpoints_and_orchestrator_share_process_local_services():
    from app.core.runtime import (
        get_extraction_orchestrator,
        get_figure_table_service,
        get_ocr_service,
        get_stamp_signature_service,
    )

    orchestrator = get_extraction_orchestrator()
    assert orchestrator.ocr is get_ocr_service()
    assert orchestrator.figure_table is get_figure_table_service()
    assert orchestrator.stamp_signature is get_stamp_signature_service()
