from unittest.mock import Mock

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.deepseek_ocr_vlm_backend import (
    DeepSeekOCRVLMBackend, VLMQualityError, column_regions,
)
from app.modules.ocr.service import OCRService
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs


def service():
    return OCRService(Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir=""))


def line(text="متن", x=0):
    return OCRLine(text=text, confidence=.9, bbox=BBox(x1=x, y1=0, x2=x+100, y2=30))


def test_quality_failure_uses_regions_and_retains_vlm_provenance(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("length")))
    monkeypatch.setattr(s._backend, "predict_regions", Mock(return_value=[line("متن صحیح " * 20)]))
    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))
    assert lines[0].text.startswith("متن صحیح")
    assert metadata.backend == "ollama_vlm"
    assert metadata.object_metadata["ocr_recovery"] == "column_regions"
    assert "ocr_recovered_by_regions" in warnings


def test_failed_regions_use_classic_provenance(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("length")))
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=VLMQualityError("empty")))
    s._fallback = Mock()
    s._fallback.predict.return_value = [line()]
    s._fallback.metadata = OCRBackendMetadata(backend="paddle", model_id="arabic", detector_id="det")
    lines, metadata, warnings = s._predict(Image.new("RGB", (900, 600)))
    assert metadata.backend == "paddle"
    assert metadata.object_metadata["text_extraction_mode"] == "ocr"
    assert "ocr_recovered_by_paddle" in warnings


@pytest.mark.parametrize("detected, succeeds", [(0, True), (2, False)])
def test_empty_classic_result_requires_no_detected_regions(monkeypatch, detected, succeeds):
    s = service()
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("empty")))
    monkeypatch.setattr(s._backend, "predict_regions", Mock(side_effect=VLMQualityError("empty")))
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
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=RuntimeError("connection refused")))
    with pytest.raises(RuntimeError, match="connection refused"):
        s._predict(Image.new("RGB", (900, 600)))


def test_regions_cover_every_pixel_once_in_rtl_order():
    boxes = column_regions(Image.new("RGB", (901, 600), "white"))
    assert boxes[0][2] == 901
    assert boxes[-1][0] == 0
    assert boxes[0][0] == boxes[1][2]
    assert boxes[1][0] == boxes[2][2]
    assert sum(b[2] - b[0] for b in boxes) == 901


def test_region_geometry_and_paragraph_reading_order(monkeypatch):
    b = DeepSeekOCRVLMBackend(Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir=""))
    responses = iter(["ستون راست", "ستون میانی", "ستون چپ"])
    monkeypatch.setattr(b, "_request", lambda image, **kwargs: next(responses))
    lines = b.predict_regions(Image.new("RGB", (900, 600), "white"))
    assert [l.bbox.x1 for l in lines] == [600, 300, 0]
    assert [p.text for p in group_lines_into_paragraphs(lines)] == ["ستون راست", "ستون میانی", "ستون چپ"]


def test_region_failure_never_returns_partial_page(monkeypatch):
    b = DeepSeekOCRVLMBackend(Settings(_env_file=None, text_extraction_mode="vlm", vlm_diagnostics_dir=""))
    responses = iter(["ستون راست", "", "ستون چپ"])
    monkeypatch.setattr(b, "_request", lambda image, **kwargs: next(responses))
    with pytest.raises(VLMQualityError):
        b.predict_regions(Image.new("RGB", (900, 600), "white"))


def test_sparse_region_output_uses_classic_verification(monkeypatch):
    s = service()
    monkeypatch.setattr(s._backend, "predict", Mock(side_effect=VLMQualityError("length")))
    monkeypatch.setattr(s._backend, "predict_regions", Mock(return_value=[line("دریای خزر")]))
    s._fallback = Mock()
    s._fallback.predict.return_value = [line("خلیج فارس")]
    s._fallback.metadata = OCRBackendMetadata(backend="paddle", model_id="arabic")
    assert s._predict(Image.new("RGB", (900, 600)))[1].backend == "paddle"
