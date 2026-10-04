from app.artifacts.publisher import ArtifactPublisher
from app.modules.ocr.bina_rizeh_backend import logical_persian_text
from app.modules.ocr.text_normalization import (
    normalize_ocr_text,
    normalize_persian_ocr_text,
    visual_persian_to_logical,
)
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from ui.text_rendering import bidi_safe_text_html, canonical_ocr_display_text


def _paragraph(*, text: str, raw_text: str) -> DetectedObject:
    return DetectedObject(
        object_id="paragraph-1",
        document_id="doc-1",
        page_id="doc-1:p1",
        page_number=1,
        type=ObjectType.PARAGRAPH,
        bbox=BBox(x1=0, y1=0, x2=100, y2=20),
        confidence=0.9,
        text=text,
        raw_text=raw_text,
        provenance=Provenance(
            module=ModuleName.OCR,
            backend="bina_rizeh",
            model_id="test",
        ),
    )


def test_bina_visual_to_logical_preserves_ascii_ltr_runs_and_reverses_persian_digits():
    assert visual_persian_to_logical("مالسOpenAI123") == "OpenAI123سلام"
    assert visual_persian_to_logical("۴۵۳۱ بوصم") == "مصوب ۱۳۵۴"
    assert visual_persian_to_logical(":تاعالطا") == "اطلاعات:"
    assert visual_persian_to_logical("388بوصم") == "مصوب388"


def test_bina_logical_text_is_canonical_persian_without_touching_raw_model_text():
    raw = "١٤٠٥ لاس رد ك ي مالس"
    logical = logical_persian_text(raw)

    assert logical == "سلام ی ک در سال ۵۰۴۱"
    assert raw == "١٤٠٥ لاس رد ك ي مالس"


def test_normalization_is_conservative_and_bidi_control_free():
    value = "  آیین\u200f  نامه  ،  دسترسی\u200c آزاد  "
    assert normalize_ocr_text(value) == "آیین نامه، دسترسی\u200cآزاد"
    assert normalize_persian_ocr_text("ك ي ١٢٣") == "ک ی ۱۲۳"


def test_ui_prefers_canonical_text_and_renders_bidi_safe_escaped_html():
    paragraphs = [
        {
            "text": "آیین‌نامه اجرایی قانون انتشار و دسترسی آزاد به اطلاعات",
            "raw_text": "تاعالطا هب دازآ یسرتسد و راشتنا نوناق ییارجا همان‌نیآ",
        }
    ]

    canonical = canonical_ocr_display_text(paragraphs)
    assert canonical.startswith("آیین‌نامه اجرایی")
    assert "تاعالطا هب" not in canonical

    rendered = bidi_safe_text_html(canonical + "\n<script>alert(1)</script>")
    assert 'dir="auto"' in rendered
    assert "unicode-bidi: plaintext" in rendered
    assert "<br>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_plain_text_artifacts_prefer_canonical_logical_text_not_visual_raw_text():
    obj = _paragraph(
        text="  آیین‌نامه  اجرایی ، قانون  انتشار  ",
        raw_text="راشتنا نوناق ،ییارجا همان‌نیآ",
    )

    output = ArtifactPublisher._ocr_plain_text([obj])
    assert output == "آیین‌نامه اجرایی، قانون انتشار\n"
    assert "راشتنا" not in output
    assert obj.raw_text == "راشتنا نوناق ،ییارجا همان‌نیآ"
