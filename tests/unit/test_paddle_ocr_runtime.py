import sys
import types

from app.core.config import Settings
from app.modules.ocr.paddle_backend import PaddleOCRBackend


def _install_fake_paddleocr(monkeypatch, captured: dict):
    module = types.ModuleType("paddleocr")

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)


def test_paddle_backend_disables_mkldnn_by_default(monkeypatch):
    captured: dict = {}
    _install_fake_paddleocr(monkeypatch, captured)

    settings = Settings(_env_file=None, ocr_backend="paddle")
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
        ocr_enable_mkldnn=True,
    )
    backend = PaddleOCRBackend(settings)
    backend._load()

    assert captured["enable_mkldnn"] is True
