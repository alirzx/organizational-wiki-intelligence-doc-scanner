from __future__ import annotations

import base64

import pytest
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import create_ocr_backend
from app.modules.ocr.deepseek_ocr_vlm_backend import (
    DeepSeekOCRVLMBackend,
    parse_deepseek_grounding_output,
    parse_deepseek_output,
)
from app.modules.ocr.paddle_backend import PaddleOCRBackend


def test_vlm_defaults_can_be_selected_explicitly():
    settings = Settings(
        _env_file=None,
        text_extraction_mode="vlm",
        vlm_backend="ollama",
        vlm_model_id="deepseek-ocr:latest",
    )
    assert settings.text_extraction_mode == "vlm"
    assert settings.vlm_backend == "ollama"
    assert settings.vlm_model_id == "deepseek-ocr:latest"


def test_factory_can_switch_between_vlm_and_classic_ocr():
    vlm_settings = Settings(
        _env_file=None,
        text_extraction_mode="vlm",
        vlm_backend="ollama",
        vlm_model_id="deepseek-ocr:latest",
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


def test_plain_text_output_uses_full_page_fallback_and_cleans_model_prefix():
    content = (
        r"\</im_start><\im_end><br>" "\n\n"
        "هیأت وزیران در جلسه ۱۳۹۴ تشکیل شد.\n\n"
        "ماده ۱- متن آیین نامه"
    )
    lines = parse_deepseek_output(content, image_width=880, image_height=1251)

    assert len(lines) == 1
    assert lines[0].text.startswith("هیأت وزیران")
    assert "im_start" not in lines[0].text
    assert lines[0].raw_text.startswith("هیأت وزیران")
    assert lines[0].confidence == 0.0
    assert lines[0].bbox.x1 == 0
    assert lines[0].bbox.y1 == 0
    assert lines[0].bbox.x2 == 880
    assert lines[0].bbox.y2 == 1251
    assert lines[0].polygon is None


def test_plain_text_output_cleans_production_tokenizer_artifacts():
    content = (
        "*<|im_end|>  هیأت وزیران در جلسه ۱۳۹۴ تشکیل شد.\n\n"
        "<|im_start|><br>ماده ۱- متن آیین نامه<|im_end|>"
    )
    lines = parse_deepseek_output(content, image_width=100, image_height=200)

    assert len(lines) == 1
    assert lines[0].text == (
        "هیأت وزیران در جلسه ۱۳۹۴ تشکیل شد.\n\n"
        "ماده ۱- متن آیین نامه"
    )
    assert "<|im_end|>" not in lines[0].text
    assert "<|im_start|>" not in lines[0].text
    assert lines[0].raw_text == lines[0].text


def test_plain_text_output_reduces_markdown_to_canonical_plain_text():
    content = (
        "```markdown\n"
        "# عنوان سند\n\n"
        "* بند اول\n"
        "+ بند دوم با **تأکید**\n"
        "```"
    )
    lines = parse_deepseek_output(content, image_width=100, image_height=200)

    assert lines[0].text == (
        "عنوان سند\n\n"
        "- بند اول\n"
        "- بند دوم با تأکید"
    )
    assert "```" not in lines[0].text
    assert "**" not in lines[0].text


def test_ollama_backend_posts_image_to_chat_endpoint(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "done": True,
                "done_reason": "stop",
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
        vlm_model_id="deepseek-ocr:latest",
        vlm_base_url="http://ollama.test:11434/",
        vlm_timeout_seconds=123,
        vlm_prompt="<|grounding|>OCR this image.",
    )
    backend = DeepSeekOCRVLMBackend(settings)
    lines = backend.predict(Image.new("RGB", (320, 200), "white"))

    assert captured["url"] == "http://ollama.test:11434/api/chat"
    assert captured["timeout"] == (10.0, 123.0)
    assert captured["json"]["model"] == "deepseek-ocr:latest"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["content"] == "<|grounding|>OCR this image."
    encoded = captured["json"]["messages"][0]["images"][0]
    assert base64.b64decode(encoded).startswith(b"\x89PNG")
    assert [line.text for line in lines] == ["متن"]
    assert backend.metadata.object_metadata["confidence_available"] is False
    assert backend.metadata.object_metadata["confidence_semantics"] == "unavailable_sentinel_zero"
    assert backend.metadata.object_metadata["geometry_available"] == "response_dependent"


@pytest.mark.parametrize("content", ["", "abcdefghij" * 200, r"\</imstart> <\|j||>"])
def test_rejects_empty_or_looping_output(content):
    with pytest.raises(RuntimeError):
        parse_deepseek_output(content, image_width=100, image_height=100)


def test_short_repeated_form_labels_are_preserved():
    content = "نام و نام خانوادگی: ______\n" * 20
    assert parse_deepseek_output(content, image_width=100, image_height=100)[0].text


@pytest.mark.parametrize("bad_response", [
    {"done": False, "message": {"content": "partial"}},
    {"done": True, "done_reason": "length", "message": {"content": "partial"}},
    {"done": True, "message": {"content": "abcdefghij" * 200}},
    {"done": True, "message": {"content": ""}},
])
def test_invalid_generation_retries_fresh_image_once(monkeypatch, bad_response):
    requests = []
    responses = iter([bad_response, {"done": True, "message": {"content": "متن صحیح"}}])

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return next(responses)

    def post(url, *, json, timeout):
        requests.append(json)
        return Response()

    monkeypatch.setattr("app.modules.ocr.deepseek_ocr_vlm_backend.requests.post", post)
    backend = DeepSeekOCRVLMBackend(Settings(_env_file=None))
    assert backend.predict(Image.new("RGB", (100, 100)))[0].text == "متن صحیح"
    assert len(requests) == 2
    assert requests[0]["options"]["num_predict"] == 4096
    assert requests[0]["options"]["num_ctx"] == 8192
    assert requests[0]["options"]["repeat_penalty"] == 1.1
    assert requests[0]["messages"][0]["content"] == "\nExtract the text in the image."
    assert requests[1]["messages"][0]["content"] == "\nFree OCR."
    assert len(requests[1]["messages"]) == 1
    assert requests[0]["messages"][0]["images"] == requests[1]["messages"][0]["images"]


def test_quality_retry_is_bounded(monkeypatch):
    calls = []
    backend = DeepSeekOCRVLMBackend(Settings(_env_file=None))

    def request(image, *, prompt=None):
        calls.append(prompt)
        return "abcdefghij" * 200

    monkeypatch.setattr(backend, "_request", request)
    with pytest.raises(RuntimeError, match="repeated"):
        backend.predict(Image.new("RGB", (100, 100)))
    assert len(calls) == 2


def test_compose_literal_newline_is_decoded():
    assert Settings(_env_file=None, vlm_prompt=r"\nFree OCR.").vlm_prompt == "\nFree OCR."


def test_malformed_production_header_is_removed():
    content = "\\</imstart>\n<\\|u{>\nday\n---\n\nهیأت وزیران در جلسه"
    assert parse_deepseek_output(content, image_width=100, image_height=200)[0].text == "هیأت وزیران در جلسه"


def test_margin_crop_preserves_ink_and_removes_outer_frame():
    from PIL import ImageDraw
    from app.modules.ocr.deepseek_ocr_vlm_backend import crop_document_margins

    image = Image.new("RGB", (400, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 385, 585), outline="gray", width=2)
    draw.rectangle((100, 100, 300, 200), fill="black")
    cropped, x, y = crop_document_margins(image)
    assert (x, y) == (76, 76)
    assert cropped.size == (249, 149)
    assert cropped.tobytes() == image.crop((76, 76, 325, 225)).tobytes()


def test_cropped_grounding_coordinates_are_restored(monkeypatch):
    from PIL import ImageDraw

    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 300, 200), fill="black")
    backend = DeepSeekOCRVLMBackend(Settings(_env_file=None))
    monkeypatch.setattr(backend, "_request", lambda image, **kwargs:
        "<|ref|>متن<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|>")
    line = backend.predict(image)[0]
    assert (line.bbox.x1, line.bbox.y1, line.bbox.x2, line.bbox.y2) == (76, 76, 325, 225)
    assert line.polygon.points[0].x == 76
    assert line.polygon.points[0].y == 76


def test_dense_page_retains_full_page_context():
    from PIL import ImageDraw
    from app.modules.ocr.deepseek_ocr_vlm_backend import crop_document_margins

    image = Image.new("RGB", (400, 600), "white")
    ImageDraw.Draw(image).rectangle((30, 40, 370, 560), fill="black")
    cropped, x, y = crop_document_margins(image)
    assert cropped is image
    assert (x, y) == (0, 0)


@pytest.mark.parametrize("content", ["day\n[[1]]\nمتن", "متن\u200c" * 50])
def test_damaged_order_or_residual_model_header_is_rejected(content):
    with pytest.raises(RuntimeError, match="damaged"):
        parse_deepseek_output(content, image_width=100, image_height=200)


@pytest.mark.parametrize("content", ["1", r"\modifier>"])
def test_page_number_only_or_malformed_artifact_is_rejected(content):
    with pytest.raises(RuntimeError):
        parse_deepseek_output(content, image_width=100, image_height=100)
