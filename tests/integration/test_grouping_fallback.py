import json
from io import BytesIO

import pytest
from PIL import Image

from app.core.config import Settings
from app.schemas.extraction import GroupingDiagnostics
from app.orchestration.extractor import ExtractionOrchestrator
from app.preprocessing.pipeline import prepare_page


def test_disabled_and_fallback_diagnostics_are_additive_and_private():
    disabled = GroupingDiagnostics()
    fallback = GroupingDiagnostics(mode="heuristic_fallback", fallback_reason="GroupingModelUnavailable")
    encoded = json.dumps(fallback.model_dump(mode="json"))
    assert disabled.mode.value == "heuristic_disabled"
    assert fallback.mode.value == "heuristic_fallback"
    assert "embedding" in fallback.model_dump() and "vector" not in encoded
    assert Settings(grouping_enabled=False).grouping_enabled is False


@pytest.mark.asyncio
async def test_missing_model_preserves_heuristic_objects():
    def page(settings):
        image = Image.new("RGB", (200, 300), "white"); buffer = BytesIO(); image.save(buffer, "PNG")
        return prepare_page(data=buffer.getvalue(), filename="p.png", mime_type="image/png", document_id="d", page_id="d:p1", page_number=1, page_metadata={}, settings=settings)
    base = Settings(_env_file=None, ocr_backend="mock", figure_table_backend="mock", stamp_signature_backend="mock", grouping_enabled=False)
    learned = Settings(_env_file=None, ocr_backend="mock", figure_table_backend="mock", stamp_signature_backend="mock", grouping_enabled=True, grouping_backend="lightgbm", grouping_model_path="missing-package")
    baseline = await ExtractionOrchestrator(base).extract_document(document_id="d", pages=[page(base)], request_id="r1", document_metadata={})
    fallback = await ExtractionOrchestrator(learned).extract_document(document_id="d", pages=[page(learned)], request_id="r2", document_metadata={})
    project = lambda response: [(obj.type, obj.text, obj.bbox.model_dump()) for obj in response.objects]
    assert project(fallback) == project(baseline)
    assert fallback.grouping.mode.value == "heuristic_fallback"


@pytest.mark.asyncio
async def test_clustering_backend_runs_without_trained_model(tmp_path):
    pytest.importorskip("sklearn")
    settings = Settings(_env_file=None, ocr_backend="mock", figure_table_backend="mock", stamp_signature_backend="mock",
                        grouping_enabled=True, grouping_backend="clustering",
                        grouping_model_path=str(tmp_path / "no-model"))
    image = Image.new("RGB", (200, 300), "white")
    buffer = BytesIO(); image.save(buffer, "PNG")
    page = prepare_page(data=buffer.getvalue(), filename="p.png", mime_type="image/png",
                        document_id="d", page_id="d:p1", page_number=1, page_metadata={}, settings=settings)
    response = await ExtractionOrchestrator(settings).extract_document(
        document_id="d", pages=[page], request_id="r", document_metadata={})
    assert response.grouping.mode.value == "clustered"
    assert response.grouping.model_package_id.startswith("dbscan.v1:")
    assert response.grouping.fallback_reason is None
    assert response.content_groups
