from app.artifacts.publisher import ArtifactPublisher
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName


def paragraph(*, object_id: str, text: str, y1: float, order: int | None) -> DetectedObject:
    metadata = {"ocr_reading_order": order} if order is not None else {}
    return DetectedObject(
        object_id=object_id,
        document_id="doc",
        page_id="doc:p1",
        page_number=1,
        type=ObjectType.PARAGRAPH,
        bbox=BBox(x1=10, y1=y1, x2=500, y2=y1 + 50),
        confidence=0.0,
        text=text,
        raw_text=text,
        metadata=metadata,
        provenance=Provenance(
            module=ModuleName.OCR,
            backend="ollama_vlm",
            model_id="deepseek-ocr:latest",
        ),
    )


def test_plain_ocr_text_uses_explicit_page_local_ocr_order():
    objects = [
        paragraph(object_id="bottom", text="بخش پایین", y1=600, order=2),
        paragraph(object_id="top", text="بخش بالا", y1=0, order=0),
        paragraph(object_id="middle", text="بخش میانی", y1=300, order=1),
    ]

    text = ArtifactPublisher._ocr_plain_text(objects)

    assert text == "بخش بالا\n\nبخش میانی\n\nبخش پایین\n"


def test_unified_layout_order_remains_geometry_authoritative():
    # OCR-internal order is deliberately ignored by the unified layout sorter.
    # layout.v2 still interleaves OCR and visual objects by their page geometry.
    top_visual = {
        "bbox": {"x1": 20, "y1": 50, "x2": 200, "y2": 150},
        "metadata": {},
        "type": "figure",
        "object_id": "figure-top",
    }
    lower_ocr = {
        "bbox": {"x1": 20, "y1": 400, "x2": 700, "y2": 500},
        "metadata": {"ocr_reading_order": 0},
        "type": "paragraph",
        "object_id": "paragraph-lower",
    }

    ordered = sorted([lower_ocr, top_visual], key=ArtifactPublisher._reading_order_key)

    assert [item["object_id"] for item in ordered] == ["figure-top", "paragraph-lower"]
