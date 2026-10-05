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


def lines_to_detected_objects(
    lines: list[OCRLine],
    *,
    page: PreparedPage,
    settings: Settings,
    backend_metadata: OCRBackendMetadata,
) -> list[DetectedObject]:
    paragraphs = group_lines_into_paragraphs(
        lines,
        max_gap_ratio=settings.ocr_paragraph_max_gap_ratio,
        min_x_overlap=settings.ocr_paragraph_min_x_overlap,
    )
    objects: list[DetectedObject] = []
    for paragraph in paragraphs:
        source_bbox = restore_bbox_to_source(
            paragraph.bbox,
            page.transform,
            page.image_metadata.source_width,
            page.image_metadata.source_height,
        )
        source_polygon = restore_polygon_to_source(
            paragraph.polygon,
            page.transform,
            page.image_metadata.source_width,
            page.image_metadata.source_height,
        ) if paragraph.polygon else None
        objects.append(
            DetectedObject(
                object_id=new_object_id("paragraph"),
                document_id=page.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                type=ObjectType.PARAGRAPH,
                bbox=source_bbox,
                polygon=source_polygon,
                confidence=paragraph.confidence,
                text=paragraph.text,
                raw_text=paragraph.raw_text,
                metadata={
                    "line_count": len(paragraph.lines),
                    **({"ocr_reading_order": paragraph.lines[0].reading_order}
                       if paragraph.lines[0].reading_order is not None else {}),
                    "line_confidences": [round(line.confidence, 6) for line in paragraph.lines],
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
        )
    return objects
