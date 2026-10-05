from dataclasses import dataclass

from app.schemas.common import BBox, Polygon


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    bbox: BBox
    polygon: Polygon | None = None
    raw_text: str | None = None
    reading_order: int | None = None


@dataclass(frozen=True)
class OCRParagraph:
    text: str
    raw_text: str
    confidence: float
    bbox: BBox
    polygon: Polygon | None
    lines: tuple[OCRLine, ...]
