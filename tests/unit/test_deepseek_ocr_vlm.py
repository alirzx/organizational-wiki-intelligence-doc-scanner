from __future__ import annotations

import base64

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import create_ocr_backend
from app.modules.ocr.deepseek_ocr_vlm_backend import (
    DeepSeekOCRVLMBackend,
    parse_deepseek_grounding_output,
)
from app.modules.ocr.paddle_backend import PaddleOCRBackend


def test_vlm_defaults_can_be_selected_explicitly():
    settings = Settings(
        _env_file=None,
        text_extraction_mode="vlm",
        vlm_backend="ollama",
        vlm_model_id="deepseek-ocr:3b",
    )
    assert settings.text_extraction_mode == "vlm"
    assert settings.vlm_backend == "ollama"
    assert settings.vlm_model_id == "deepseek-ocr:3b"


def test_factory_can_switch_between_vlm_and_classic_ocr():
    vlm_settings = Settings(
        _env_file=None,
        text_extraction_mode="vlm",
        vlm_backend="ollama",
        vlm_model_id="deepseek-ocr:3b",
    )
    assert isinstance(create_ocr_backend(vlm_settings), DeepSeekOCRVLMBackend)

    ocr_settings = Settings(
        _env_file=None,
        text_extraction_mode="ocr",
        ocr_backend="paddle",
        ocr_device="cpu",
    )
    assert isinstance(create_ocr_backend(ocr_settings), PaddleOCRBackend)


def test_grounding_parser_maps_coordinates_and_normalizes_persian():
    content = (
        "<|ref|>سلام دنيا<|/ref|><|det|>[[100, 200, 900, 260]]<|/det|>\n"
        "<|ref|>OpenAI-123<|/ref|><|det|>[[700, 300, 950, 350]]<|/det|>"
    )
    lines = parse_deepseek_grounding_output(
        content,
        image_width=1000,
        image_height=2000,
    )

    assert [line.text for line in lines] == ["سلام دنیا", "OpenAI-123"]
    assert lines[0].raw_text == "سلام دنيا"
    assert lines[0].confidence == 0.0
    assert lines[0].bbox.x1 == pytest.approx(100 / 999 * 1000)
    assert lines[0].bbox.y1 == pytest.approx(200 / 999 * 2000)
    assert lines[0].polygon is not None


def test_grounding_parser_rejects_non_grounded_response():
    with pytest.raises(RuntimeError, match="grounding tags"):
        parse_deepseek_grounding_output(
            "plain OCR without positions",
            image_width=100,
            image_height=100,
        )


def test_ollama_backend_posts_image_to_chat_endpoint(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        "<|ref|>متن<|/ref|>"
                        "<|det|>[[100, 100, 800, 200]]<|/det|>"
                    )
                }
            }

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("app.modules.ocr.deepseek_ocr_vlm_backend.requests.post", fake_post)
    settings = Settings(
        _env_file=None,
        text_extraction_mode="vlm",
        vlm_backend="ollama",
        vlm_model_id="deepseek-ocr:3b",
        vlm_base_url="http://ollama.test:11434/",
        vlm_timeout_seconds=123,
        vlm_prompt="<|grounding|>OCR this image.",
    )
    backend = DeepSeekOCRVLMBackend(settings)
    lines = backend.predict(Image.new("RGB", (320, 200), "white"))

    assert captured["url"] == "http://ollama.test:11434/api/chat"
    assert captured["timeout"] == (10.0, 123.0)
    assert captured["json"]["model"] == "deepseek-ocr:3b"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["content"] == "<|grounding|>OCR this image."
    encoded = captured["json"]["messages"][0]["images"][0]
    assert base64.b64decode(encoded).startswith(b"\x89PNG")
    assert [line.text for line in lines] == ["متن"]
    assert backend.metadata.object_metadata["confidence_available"] is False
    assert backend.metadata.object_metadata["confidence_semantics"] == "unavailable_sentinel_zero"
