from io import BytesIO

from PIL import Image

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.types import OCRLine
from app.modules.ocr.vlm_postprocessing import clean_vlm_canonical_text, clean_vlm_lines
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


def test_vlm_cleanup_preserves_raw_text_for_debugging():
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
    assert cleaned.raw_text == raw


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
