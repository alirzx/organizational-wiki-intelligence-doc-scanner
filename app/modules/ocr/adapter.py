from __future__ import annotations

from app.core.config import Settings
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.types import OCRLine
from app.preprocessing.transforms import restore_bbox_to_source, restore_polygon_to_source
from app.preprocessing.types import PreparedPage
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.text_processing.ids import stable_block_id
from app.text_processing.normalizer import TextNormalizer
from app.text_processing.types import ContentGroup, TextBlock
from app.utils.ids import new_object_id


def lines_to_text_blocks(
    lines: list[OCRLine],
    *,
    page: PreparedPage,
    normalizer: TextNormalizer | None = None,
) -> list[TextBlock]:
    """Convert OCR lines to stable source-coordinate atomic blocks."""

    text_normalizer = normalizer or TextNormalizer()
    ordered = sorted(lines, key=lambda line: (line.bbox.y1, line.bbox.x1, line.text))
    blocks: list[TextBlock] = []
    seen_ids: set[str] = set()
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
        normalized = text_normalizer.normalize(line.text)
        collision = 0
        block_id = stable_block_id(
            document_id=page.document_id,
            page_id=page.page_id,
            bbox=source_bbox,
            content_ordinal=ordinal,
            text=line.text,
        )
        while block_id in seen_ids:
            collision += 1
            block_id = stable_block_id(
                document_id=page.document_id,
                page_id=page.page_id,
                bbox=source_bbox,
                content_ordinal=ordinal,
                text=line.text,
                collision=collision,
            )
        seen_ids.add(block_id)
        blocks.append(
            TextBlock(
                block_id=block_id,
                document_id=page.document_id,
                page_id=page.page_id,
                page_number=page.page_number,
                content_ordinal=ordinal,
                original_text=normalized.original,
                normalized_text=normalized.normalized,
                bbox=source_bbox,
                polygon=source_polygon,
                ocr_confidence=line.confidence,
                page_width=page.image_metadata.source_width,
                page_height=page.image_metadata.source_height,
                metadata={"normalizer_version": normalized.profile_version},
            )
        )
    return blocks


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
