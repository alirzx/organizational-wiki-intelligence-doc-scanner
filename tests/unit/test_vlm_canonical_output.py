from io import BytesIO

from PIL import Image

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.types import OCRLine
from app.modules.ocr.vlm_postprocessing import (
    VLM_ARTIFACT_DICTIONARY,
    clean_vlm_canonical_text,
    clean_vlm_lines,
    strip_vlm_artifacts,
)
from app.preprocessing.pipeline import prepare_page
from app.schemas.common import BBox


def _page():
    image = Image.new("RGB", (900, 1200), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return prepare_page(
        data=buffer.getvalue(),
        filename="page.png",
        mime_type="image/png",
        document_id="doc",
        page_id="doc:p1",
        page_number=1,
        page_metadata={},
        settings=Settings(_env_file=None, preprocess_max_long_edge=1200),
    )


def test_residual_deepseek_role_tokens_are_removed_from_canonical_text():
    value = "</|assistant>\n\\ |Im_begin_block|>\n2- مؤسسات مشمول قانون"
    assert clean_vlm_canonical_text(value) == "2- مؤسسات مشمول قانون"


def test_vlm_cleanup_removes_artifacts_from_persisted_raw_and_canonical_text():
    raw = "</|assistant>\n\\ |Im_begin_block|>\nمتن"
    line = OCRLine(
        text=raw,
        raw_text=raw,
        confidence=0.0,
        bbox=BBox(x1=0, y1=0, x2=900, y2=400),
        reading_order=0,
    )
    cleaned = clean_vlm_lines([line])[0]
    assert cleaned.text == "متن"
    assert cleaned.raw_text == "متن"


def test_all_known_control_token_families_have_a_dictionary_entry():
    assert "im_start" in VLM_ARTIFACT_DICTIONARY["image_transport"]
    assert "im_continue" in VLM_ARTIFACT_DICTIONARY["image_transport"]
    assert "assistant" in VLM_ARTIFACT_DICTIONARY["roles"]
    assert "ref" in VLM_ARTIFACT_DICTIONARY["grounding"]
    assert "m_end" in VLM_ARTIFACT_DICTIONARY["legacy_pipe"]


def test_real_document_transport_variants_are_removed_but_persian_text_survives():
    value = """
</|im_start|>user
<|im_continue|
\\ |im_end|>
\\ |m| client
\\ |c|
\\ |m_End|
0</|im_start|>|user|l3</|ref|>
مدیریت عملیات
"""
    cleaned = clean_vlm_canonical_text(value)
    assert cleaned == "مدیریت عملیات"
    for artifact in ("im_start", "im_continue", "im_end", "user", "client", "|m|", "|c|"):
        assert artifact not in cleaned.lower()


def test_image_description_blocks_are_removed_from_ocr_content():
    value = """
</|im_start|>
The image displays a blue background with two tall buildings in the center.
The style of the image is 3D render.

</|im_start|>
This is a photograph of a large white building with a curved roof and a fence around it.
In the background, there are two snow-capped mountains and a clear blue sky.

مدیریت تامین و توزیع
"""
    assert clean_vlm_canonical_text(value) == "مدیریت تامین و توزیع"


def test_generated_visual_caption_without_image_word_is_removed():
    value = """
This is a black and white line drawing of a hand making an 'okay' gesture. The fingers are spread apart.

فهرست
"""
    assert clean_vlm_canonical_text(value) == "فهرست"


def test_embedded_base64_and_pdf_tool_metadata_are_removed():
    value = """
assistant:data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB
< 用户
用户
function="get_pdf_title" />
function="get_pdf_author" />
[DOCUMENT]
متن اصلی
"""
    cleaned = clean_vlm_canonical_text(value)
    assert "data:image" not in cleaned
    assert "get_pdf_" not in cleaned
    assert "DOCUMENT" not in cleaned
    assert "assistant" not in cleaned.lower()
    assert "متن اصلی" in cleaned


def test_html_presentation_tags_are_removed_but_table_text_is_kept():
    value = "<table><td>کمیسیون مناقصات</td><td>بازرسی فنی</td></table>"
    cleaned = strip_vlm_artifacts(value)
    assert "<table" not in cleaned
    assert "<td" not in cleaned
    assert "کمیسیون مناقصات" in cleaned
    assert "بازرسی فنی" in cleaned


def test_artifact_only_vlm_line_is_dropped_and_reading_order_compacted():
    lines = [
        OCRLine(
            "مدیریت عملیات",
            0.0,
            BBox(x1=0, y1=0, x2=900, y2=300),
            raw_text="مدیریت عملیات",
            reading_order=0,
        ),
        OCRLine(
            "</|im_start|> This image displays a blue sky with white clouds.",
            0.0,
            BBox(x1=0, y1=300, x2=900, y2=600),
            raw_text="</|im_start|> This image displays a blue sky with white clouds.",
            reading_order=1,
        ),
        OCRLine(
            "مدیریت مالی",
            0.0,
            BBox(x1=0, y1=600, x2=900, y2=900),
            raw_text="مدیریت مالی",
            reading_order=2,
        ),
    ]

    cleaned = clean_vlm_lines(lines)

    assert [line.text for line in cleaned] == ["مدیریت عملیات", "مدیریت مالی"]
    assert [line.reading_order for line in cleaned] == [0, 1]


def test_top_to_bottom_regions_remain_separate_layout_paragraphs():
    lines = [
        OCRLine("بالا", 0.0, BBox(x1=0, y1=0, x2=900, y2=400), reading_order=0),
        OCRLine("میانه", 0.0, BBox(x1=0, y1=400, x2=900, y2=800), reading_order=1),
        OCRLine("پایین", 0.0, BBox(x1=0, y1=800, x2=900, y2=1200), reading_order=2),
    ]
    metadata = OCRBackendMetadata(
        backend="ollama_vlm",
        model_id="deepseek-ocr:latest",
        object_metadata={
            "text_extraction_mode": "vlm",
            "ocr_strategy": "top_to_bottom_regions",
        },
    )

    objects = lines_to_detected_objects(
        lines,
        page=_page(),
        settings=Settings(_env_file=None),
        backend_metadata=metadata,
    )

    assert [obj.text for obj in objects] == ["بالا", "میانه", "پایین"]
    assert [obj.metadata["ocr_reading_order"] for obj in objects] == [0, 1, 2]
    assert [obj.bbox.y1 for obj in objects] == [0, 400, 800]
    assert [obj.bbox.y2 for obj in objects] == [400, 800, 1200]
