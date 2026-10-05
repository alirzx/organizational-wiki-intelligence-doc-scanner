from __future__ import annotations

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.types import OCRLine
from app.preprocessing.transforms import restore_bbox_to_source, restore_polygon_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.utils.ids import new_object_id


def _detected_object_from_lines(
    lines: tuple[OCRLine, ...],
    *,
    text: str,
    raw_text: str,
    confidence: float,
    page: PreparedPage,
    backend_metadata: OCRBackendMetadata,
) -> DetectedObject:
    x1 = min(line.bbox.x1 for line in lines)
    y1 = min(line.bbox.y1 for line in lines)
    x2 = max(line.bbox.x2 for line in lines)
    y2 = max(line.bbox.y2 for line in lines)
    from app.schemas.common import BBox, Point, Polygon

    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    polygon = Polygon(
        points=[
            Point(x=x1, y=y1),
            Point(x=x2, y=y1),
            Point(x=x2, y=y2),
            Point(x=x1, y=y2),
        ]
    )
    source_bbox = restore_bbox_to_source(
        bbox,
        page.transform,
        page.image_metadata.source_width,
        page.image_metadata.source_height,
    )
    source_polygon = restore_polygon_to_source(
        polygon,
        page.transform,
        page.image_metadata.source_width,
        page.image_metadata.source_height,
    )
    return DetectedObject(
        object_id=new_object_id("paragraph"),
        document_id=page.document_id,
        page_id=page.page_id,
        page_number=page.page_number,
        type=ObjectType.PARAGRAPH,
        bbox=source_bbox,
        polygon=source_polygon,
        confidence=confidence,
        text=text,
        raw_text=raw_text,
        metadata={
            "line_count": len(lines),
            **(
                {"ocr_reading_order": lines[0].reading_order}
                if lines[0].reading_order is not None
                else {}
            ),
            "line_confidences": [round(line.confidence, 6) for line in lines],
            **(
                {
                    "text_detection_model": backend_metadata.detector_id,
                    "text_recognition_model": backend_metadata.model_id,
                }
                if backend_metadata.detector_id is not None
                else {}
            ),
            **(
                {"model_revision": backend_metadata.model_revision}
                if backend_metadata.model_revision is not None
                else {}
            ),
            **backend_metadata.object_metadata,
        },
        provenance=Provenance(
            module=ModuleName.OCR,
            backend=backend_metadata.backend,
            model_id=backend_metadata.model_id,
            model_version=backend_metadata.model_revision,
        ),
    )


def lines_to_detected_objects(
    lines: list[OCRLine],
    *,
    page: PreparedPage,
    settings: Settings,
    backend_metadata: OCRBackendMetadata,
) -> list[DetectedObject]:
    # Top-to-bottom VLM recovery deliberately gives each region its own source-space
    # geometry. Do not merge those independent regions back into one full-page
    # paragraph, otherwise layout.json loses the very reading-order information the
    # region strategy was introduced to preserve.
    if backend_metadata.object_metadata.get("ocr_strategy") == "top_to_bottom_regions":
        objects: list[DetectedObject] = []
        for line in sorted(
            lines,
            key=lambda item: (
                item.reading_order if item.reading_order is not None else float("inf"),
                item.bbox.y1,
                -item.bbox.x1,
            ),
        ):
            text = line.text.strip()
            if not text:
                continue
            raw_text = line.raw_text if line.raw_text is not None else line.text
            objects.append(
                _detected_object_from_lines(
                    (line,),
                    text=text,
                    raw_text=raw_text,
                    confidence=max(0.0, min(1.0, line.confidence)),
                    page=page,
                    backend_metadata=backend_metadata,
                )
            )
        return objects

    paragraphs = group_lines_into_paragraphs(
        lines,
        max_gap_ratio=settings.ocr_paragraph_max_gap_ratio,
        min_x_overlap=settings.ocr_paragraph_min_x_overlap,
    )
    objects: list[DetectedObject] = []
    for paragraph in paragraphs:
        objects.append(
            _detected_object_from_lines(
                paragraph.lines,
                text=paragraph.text,
                raw_text=paragraph.raw_text,
                confidence=max(0.0, min(1.0, paragraph.confidence)),
                page=page,
                backend_metadata=backend_metadata,
            )
        )
    return objects
