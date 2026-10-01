from dataclasses import dataclass

from app.schemas.common import BBox, Polygon
from app.schemas.detection import DetectedObject
from app.text_processing.types import TextBlock


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    bbox: BBox
    polygon: Polygon | None = None


@dataclass(frozen=True)
class OCRParagraph:
    text: str
    raw_text: str
    confidence: float
    bbox: BBox
    polygon: Polygon | None
    lines: tuple[OCRLine, ...]


@dataclass(frozen=True)
class OCRAnalysis:
    """Internal OCR result used by document-scoped grouping.

    This envelope is intentionally not part of ModulePageResponse serialization.
    """

    lines: tuple[OCRLine, ...]
    blocks: tuple[TextBlock, ...]
    heuristic_objects: tuple[DetectedObject, ...]
