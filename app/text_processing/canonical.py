"""Canonical source-coordinate OCR block construction shared by runtime and training."""
from app.schemas.common import BBox, Polygon
from app.text_processing.ids import stable_block_id
from app.text_processing.normalizer import TextNormalizer, NormalizedText
from app.text_processing.types import TextBlock

_NORMALIZER = TextNormalizer()


def canonical_block(*, document_id: str, page_id: str, page_number: int,
                    ordinal: int, text: str, bbox: BBox, page_width: int,
                    page_height: int, confidence: float | None = None,
                    polygon: Polygon | None = None, column_id: str | None = None,
                    block_type: str | None = None, normalizer: TextNormalizer | None = None,
                    normalization: NormalizedText | None = None) -> TextBlock:
    normalized = normalization or (normalizer or _NORMALIZER).normalize(text)
    if normalized.original != text:
        raise ValueError('precomputed normalization must correspond to the original text')
    block_id = stable_block_id(document_id=document_id, page_id=page_id, bbox=bbox,
                               content_ordinal=ordinal, text=normalized.normalized)
    return TextBlock(block_id=block_id, document_id=document_id, page_id=page_id,
                     page_number=page_number, content_ordinal=ordinal,
                     original_text=normalized.original, normalized_text=normalized.normalized,
                     bbox=bbox, page_width=page_width, page_height=page_height,
                     polygon=polygon, ocr_confidence=confidence, column_id=column_id,
                     block_type=block_type, metadata={'normalizer_version': normalized.profile_version})


def block_from_record(record: dict, document_id: str) -> TextBlock:
    return canonical_block(document_id=document_id, page_id=record['page_id'],
                           page_number=record['page_number'], ordinal=record['content_ordinal'],
                           text=record.get('original_text', record['text']), bbox=BBox(**record['bbox']),
                           page_width=record['page_width'], page_height=record['page_height'],
                           confidence=record.get('confidence'), column_id=record.get('column_id'),
                           block_type=record.get('block_type'))
