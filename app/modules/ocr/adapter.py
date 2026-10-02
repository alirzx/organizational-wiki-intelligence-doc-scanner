from __future__ import annotations

from app.core.config import Settings
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.types import OCRLine
from app.preprocessing.transforms import restore_bbox_to_source, restore_polygon_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.text_processing.normalizer import TextNormalizer
from app.text_processing.types import ContentGroup, TextBlock
from app.text_processing.canonical import canonical_block
from app.utils.ids import new_object_id


def lines_to_text_blocks(
    lines: list[OCRLine],
    *,
    page: PreparedPage,
    normalizer: TextNormalizer | None = None,
) -> list[TextBlock]:
    """Convert OCR lines to stable source-coordinate atomic blocks."""

    text_normalizer = normalizer or TextNormalizer()
    ordered = sorted(lines, key=lambda line: (line.bbox.y1, line.bbox.x1, line.bbox.y2, line.bbox.x2, line.text))
    blocks: list[TextBlock] = []
    for ordinal, line in enumerate(ordered):
        source_bbox = restore_bbox_to_source(
            line.bbox,
            page.transform,
            page.image_metadata.source_width,
            page.image_metadata.source_height,
        )
        source_polygon = (
            restore_polygon_to_source(
                line.polygon,
                page.transform,
                page.image_metadata.source_width,
                page.image_metadata.source_height,
            )
            if line.polygon
            else None
        )
        blocks.append(canonical_block(
            document_id=page.document_id, page_id=page.page_id, page_number=page.page_number,
            ordinal=ordinal, text=line.text, bbox=source_bbox, polygon=source_polygon,
            confidence=line.confidence, page_width=page.image_metadata.source_width,
            page_height=page.image_metadata.source_height, normalizer=text_normalizer,
            column_id=line.column_id, block_type=line.block_type))
    return blocks


def objects_to_text_blocks(objects: list[DetectedObject], *, page: PreparedPage) -> list[TextBlock]:
    """Compatibility adapter for providers without retained line information."""
    selected = sorted((obj for obj in objects if obj.type == ObjectType.PARAGRAPH),
                      key=lambda obj: (obj.bbox.y1, obj.bbox.x1, obj.bbox.y2, obj.bbox.x2, obj.raw_text or obj.text or ''))
    return [canonical_block(document_id=page.document_id, page_id=page.page_id,
                            page_number=page.page_number, ordinal=i, text=obj.raw_text or obj.text or '',
                            bbox=obj.bbox, page_width=page.image_metadata.source_width,
                            page_height=page.image_metadata.source_height, confidence=obj.confidence,
                            polygon=obj.polygon, column_id=obj.metadata.get('column_id'),
                            block_type=obj.metadata.get('block_type')) for i, obj in enumerate(selected)]


def lines_to_detected_objects(
    lines: list[OCRLine],
    *,
    page: PreparedPage,
    settings: Settings,
    backend_name: str,
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
                    "line_confidences": [round(line.confidence, 6) for line in paragraph.lines],
                    "text_detection_model": settings.ocr_text_detection_model_name,
                    "text_recognition_model": settings.ocr_model_id,
                },
                provenance=Provenance(
                    module=ModuleName.OCR,
                    backend=backend_name,
                    model_id=settings.ocr_model_id,
                ),
            )
        )
    return objects


def groups_to_detected_objects(
    groups: list[ContentGroup],
    *,
    document_id: str,
    backend_name: str,
    model_id: str,
) -> list[DetectedObject]:
    """Project document groups into page-local paragraph objects without a new object type."""
    objects: list[DetectedObject] = []
    for group in groups:
        for span_index, span in enumerate(group.page_spans):
            objects.append(DetectedObject(
                object_id=f"paragraph_{group.group_id}_{span.page_number}",
                document_id=document_id,
                page_id=span.page_id,
                page_number=span.page_number,
                type=ObjectType.PARAGRAPH,
                bbox=span.bbox,
                confidence=group.confidence,
                text=span.text,
                raw_text=span.raw_text,
                metadata={
                    "group_id": group.group_id,
                    "group_order": group.group_order,
                    "group_type": group.group_type.value,
                    "group_cross_page": group.cross_page,
                    "group_span_index": span_index,
                    "group_member_ids": list(span.member_ids),
                },
                provenance=Provenance(module=ModuleName.OCR, backend=backend_name, model_id=model_id),
            ))
    return objects
