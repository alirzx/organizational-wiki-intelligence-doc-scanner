from io import BytesIO
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.native_recovery import recover_native_text
from app.modules.ocr.native_text import native_text_objects, covered_area
from app.modules.ocr.service import OCRService
from app.preprocessing.pipeline import prepare_page
from app.schemas.common import BBox
from app.schemas.storage import NativeTextPage, MinioDocumentRequest


def page_and_native():
    settings = Settings(_env_file=None, ocr_backend="mock")
    buffer = BytesIO()
    Image.new("RGB", (400, 300), "white").save(buffer, "PNG")
    native = {"page_number": 1, "width": 400, "height": 300, "blocks": [
        {"text": "5056 نفر", "bbox": {"x1": 200, "y1": 20, "x2": 390, "y2": 100}},
        {"text": "متن ستون چپ", "bbox": {"x1": 10, "y1": 20, "x2": 180, "y2": 100}},
    ]}
    page = prepare_page(
        data=buffer.getvalue(), filename="native.png", mime_type="image/png",
        document_id="305", page_id="305:p1", page_number=1,
        page_metadata={"_pdf_text": native}, settings=settings,
    )
    return settings, page, native


def test_native_text_keeps_numbers_source_geometry_and_order():
    _, page, native = page_and_native()
    objects = native_text_objects(page, native, [BBox(x1=210,y1=30,x2=380,y2=50)], [])
    assert [obj.text for obj in objects] == ["5056 نفر", "متن ستون چپ"]
    assert [obj.metadata["ocr_reading_order"] for obj in objects] == [1,2]
    assert objects[0].bbox.x1 == 200
    assert all(obj.provenance.backend == "pdf_text" for obj in objects)


def test_incomplete_text_layer_does_not_hide_scanned_body():
    _, page, native = page_and_native()
    assert native_text_objects(page, native, [BBox(x1=20,y1=150,x2=350,y2=200)], []) is None


def test_native_detection_coverage_restores_preprocessing_scale():
    _, page, native = page_and_native()
    page = replace(page, transform=page.transform.model_copy(update={"scale_x": .5,"scale_y": .5}))
    objects = native_text_objects(page, native, [BBox(x1=110,y1=15,x2=180,y2=30)], [])
    assert objects is not None


def test_union_coverage_does_not_double_count_overlapping_regions():
    region = BBox(x1=0,y1=0,x2=50,y2=100)
    assert covered_area(BBox(x1=0,y1=0,x2=100,y2=100), [region,region]) == .5


@pytest.mark.parametrize("change", [{"page_number":2}, {"width":401}])
def test_native_text_identity_and_dimensions_must_match_image(change):
    _, page, native = page_and_native()
    with pytest.raises(ValueError, match="does not match"):
        native_text_objects(page, {**native,**change}, [], [])


def test_native_geometry_outside_source_page_is_rejected():
    _, _, native = page_and_native()
    native["blocks"][0]["bbox"]["x2"] = 500
    with pytest.raises(ValueError, match="inside"):
        NativeTextPage.model_validate(native)


def test_request_rejects_native_page_assigned_to_wrong_image():
    _, _, native = page_and_native()
    with pytest.raises(ValueError, match="page_number"):
        MinioDocumentRequest.model_validate({
            "document_id":"305", "pages":[{"image_url":"images/p2.jpg","page_number":2,"native_text":native}],
        })


@pytest.mark.asyncio
async def test_hybrid_recovery_preserves_selected_bina_backend_and_reports_hybrid_status():
    from app.modules.ocr.types import OCRLine

    settings, page, _ = page_and_native()
    service = OCRService(settings)

    verifier = Mock()
    verifier.predict.return_value = []
    verifier.last_detection_boxes = [BBox(x1=20,y1=150,x2=350,y2=200)]
    verifier.metadata = OCRBackendMetadata(backend="paddle", model_id="paddle")

    recovery = Mock()
    recovery.metadata = OCRBackendMetadata(backend="bina_rizeh", model_id="bina")
    recovery.predict.return_value = [OCRLine(
        text="متن اسکن شده",
        confidence=.9,
        bbox=BBox(x1=0,y1=0,x2=100,y2=30),
    )]

    service._fallback = verifier
    service._backend = recovery
    response = await service.run_with_native_text(page, "req-native", [])

    recovered = [obj for obj in response.objects if obj.provenance.backend != "pdf_text"]
    assert [obj.text for obj in recovered] == ["متن اسکن شده"]
    assert recovered[0].provenance.backend == "bina_rizeh"
    assert response.status.backend == "pdf_hybrid"
    assert "bina_rizeh" in response.status.model_id
    assert "missing_pdf_text_region_recovered_by_ocr" in response.status.warnings


def test_hybrid_recovery_deduplicates_repeated_missing_region_text():
    from app.modules.ocr.types import OCRLine

    _, page, _ = page_and_native()
    verifier = Mock()
    verifier.predict.return_value = []
    verifier.last_detection_boxes = [
        BBox(x1=20,y1=150,x2=180,y2=200),
        BBox(x1=200,y1=150,x2=350,y2=200),
    ]
    verifier.metadata = OCRBackendMetadata(backend="paddle", model_id="paddle")

    recovery = Mock()
    recovery.metadata = OCRBackendMetadata(backend="bina_rizeh", model_id="bina")
    recovery.predict.return_value = [OCRLine(
        text="متن تکراری",
        confidence=.9,
        bbox=BBox(x1=0,y1=0,x2=80,y2=30),
    )]

    objects, _ = recover_native_text(
        page, verifier, recovery, [], allow_classic_fallback=False,
    )
    assert [obj.text for obj in objects].count("متن تکراری") == 1


@pytest.mark.asyncio
async def test_unreadable_embedded_image_keeps_native_text_and_reports_warning():
    settings, page, native = page_and_native()
    native["image_regions"] = [{"x1":10,"y1":140,"x2":380,"y2":250}]

    verifier = Mock()
    verifier.predict.return_value = []
    verifier.last_detection_boxes = [BBox(x1=20,y1=150,x2=350,y2=200)]
    verifier.metadata = OCRBackendMetadata(backend="paddle", model_id="paddle")

    recovery = Mock()
    recovery.metadata = OCRBackendMetadata(backend="bina_rizeh", model_id="bina")
    recovery.predict.return_value = []

    service = OCRService(settings)
    service._fallback = verifier
    service._backend = recovery
    response = await service.run_with_native_text(page, "req-native", [])

    assert [obj.text for obj in response.objects] == ["5056 نفر", "متن ستون چپ"]
    assert response.status.backend == "pdf_text"
    assert "unreadable_embedded_image_text_preserved_as_image" in response.status.warnings


@pytest.mark.parametrize("bad_text", ["   ", "text\ufffd", "text\x01"])
def test_native_text_rejects_clear_extraction_corruption(bad_text):
    with pytest.raises(ValueError):
        NativeTextPage.model_validate({
            "page_number": 1,
            "width": 100,
            "height": 100,
            "blocks": [{"text": bad_text, "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 20}}],
        })

