from unittest.mock import Mock

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.deepseek_ocr_vlm_backend import (
    DeepSeekOCRVLMBackend,
    VLMQualityError,
    horizontal_regions,
)
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.service import OCRService
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox


def service():
    return OCRService(Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir=""))


def line(text="متن", x=0, y=0, reading_order=None):
    return OCRLine(
        text=text,
        confidence=.9,
        bbox=BBox(x1=x, y1=y, x2=x + 100, y2=y + 30),
        reading_order=reading_order,
    )


def test_primary_vlm_strategy_uses_top_to_bottom_regions(monkeypatch):
    s = service()
    monkeypatch.setattr(
        s._backend,
        "predict_regions",
        Mock(return_value=[line("بخش اول", y=0, reading_order=0), line("بخش دوم", y=200, reading_order=1)]),
    )
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=AssertionError("full page should not run")))

    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))

    assert [item.text for item in lines] == ["بخش اول", "بخش دوم"]
    assert metadata.backend == "ollama_vlm"
    assert metadata.object_metadata["ocr_strategy"] == "top_to_bottom_regions"
    assert warnings == []


def test_failed_regions_use_full_page_vlm_recovery(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=VLMQualityError("region failed")))
    monkeypatch.setattr(s._backend, "predict", Mock(return_value=[line("متن کامل")]))

    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))

    assert lines[0].text == "متن کامل"
    assert metadata.backend == "ollama_vlm"
    assert metadata.object_metadata["ocr_strategy"] == "full_page_recovery"
    assert "ocr_recovered_by_full_page_vlm" in warnings


def test_failed_vlm_paths_use_classic_provenance(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=VLMQualityError("region failed")))
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("page failed")))
    s._fallback = Mock()
    s._fallback.predict.return_value = [line()]
    s._fallback.last_detection_count = 1
    s._fallback.metadata = OCRBackendMetadata(backend="paddle", model_id="arabic", detector_id="det")

    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))

    assert metadata.backend == "paddle"
    assert metadata.object_metadata["text_extraction_mode"] == "ocr"
    assert metadata.object_metadata["ocr_strategy"] == "paddle_fallback"
    assert "ocr_recovered_by_paddle" in warnings


@pytest.mark.parametrize("detected, succeeds", [(0, True), (2, False)])
def test_empty_classic_result_requires_no_detected_regions(monkeypatch, detected, succeeds):
    s = service()
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=VLMQualityError("empty")))
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("empty")))
    s._fallback = Mock()
    s._fallback.predict.return_value = []
    s._fallback.last_detection_count = detected
    s._fallback.metadata = OCRBackendMetadata(backend="paddle", model_id="arabic")
    if succeeds:
        assert s._predict(Image.new("RGB", (900, 600)))[0] == []
    else:
        with pytest.raises(VLMQualityError, match="detected"):
            s._predict(Image.new("RGB", (900, 600)))


def test_transport_error_is_not_hidden_by_fallback(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=RuntimeError("connection refused")))
    with pytest.raises(RuntimeError, match="connection refused"):
        s._predict(Image.new("RGB", (900, 600)))


def test_regions_cover_every_pixel_once_top_to_bottom():
    boxes = horizontal_regions(Image.new("RGB", (900, 601), "white"))
    assert boxes[0][1] == 0
    assert boxes[-1][3] == 601
    assert boxes[0][3] == boxes[1][1]
    assert boxes[1][3] == boxes[2][1]
    assert all(box[0] == 0 and box[2] == 900 for box in boxes)
    assert sum(box[3] - box[1] for box in boxes) == 601


def test_region_geometry_and_paragraph_reading_order(monkeypatch):
    b = DeepSeekOCRVLMBackend(
        Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir="", vlm_crop_margins=False)
    )
    responses = iter(["بخش بالا", "بخش میانی", "بخش پایین"])
    monkeypatch.setattr(b, "_request", lambda image, **kwargs: next(responses))

    lines = b.predict_regions(Image.new("RGB", (900, 600), "white"))

    assert [item.bbox.y1 for item in lines] == [0, 200, 400]
    assert [item.reading_order for item in lines] == [0, 1, 2]
    assert [p.text for p in group_lines_into_paragraphs(lines)] == [
        "بخش بالا",
        "بخش میانی",
        "بخش پایین",
    ]


def test_region_failure_never_returns_partial_page(monkeypatch):
    b = DeepSeekOCRVLMBackend(
        Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir="", vlm_crop_margins=False)
    )
    responses = iter(["بخش بالا", "", "بخش پایین"])
    monkeypatch.setattr(b, "_request", lambda image, **kwargs: next(responses))
    with pytest.raises(VLMQualityError):
        b.predict_regions(Image.new("RGB", (900, 600), "white"))


def test_sparse_valid_region_output_is_not_replaced_by_classic_ocr(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict_regions", Mock(return_value=[line("دریای خزر", reading_order=0)]))
    s._fallback = Mock()

    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))

    assert lines[0].text == "دریای خزر"
    assert metadata.backend == "ollama_vlm"
    assert warnings == []
    s._fallback.predict.assert_not_called()
