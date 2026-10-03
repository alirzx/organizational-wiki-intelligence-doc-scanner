import sys
import types

from app.core.config import Settings
from app.modules.ocr.paddle_backend import PaddleOCRBackend


def _install_fake_paddleocr(monkeypatch, captured: dict):
    paddle = types.ModuleType("paddle")
    paddle.is_compiled_with_cuda = lambda: False
    paddle.device = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "paddle", paddle)

    module = types.ModuleType("paddleocr")

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)


def test_paddle_backend_disables_mkldnn_by_default(monkeypatch):
    captured: dict = {}
    _install_fake_paddleocr(monkeypatch, captured)

    settings = Settings(_env_file=None, ocr_backend="paddle", ocr_device="cpu")
    backend = PaddleOCRBackend(settings)
    backend._load()

    assert settings.ocr_enable_mkldnn is False
    assert captured["enable_mkldnn"] is False
    assert captured["device"] == "cpu"


def test_paddle_backend_allows_explicit_mkldnn_enable(monkeypatch):
    captured: dict = {}
    _install_fake_paddleocr(monkeypatch, captured)

    settings = Settings(
        _env_file=None,
        ocr_backend="paddle",
        ocr_device="cpu",
        ocr_enable_mkldnn=True,
    )
    backend = PaddleOCRBackend(settings)
    backend._load()

    assert captured["enable_mkldnn"] is True


def test_paddle_backend_metadata_keeps_existing_detector_and_recognizer_settings():
    settings = Settings(
        _env_file=None,
        ocr_backend="paddle",
        ocr_model_id="PaddlePaddle/custom_recognizer",
        ocr_text_detection_model_name="custom_detector",
    )

    metadata = PaddleOCRBackend(settings).metadata

    assert metadata.backend == "paddle"
    assert metadata.model_id == "PaddlePaddle/custom_recognizer"
    assert metadata.detector_id == "custom_detector"
