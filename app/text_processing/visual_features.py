from __future__ import annotations

import math

from app.text_processing.types import TextBlock


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def extract_visual_features(a: TextBlock, b: TextBlock) -> dict[str, float]:
    width_base = max(a.bbox.width, b.bbox.width)
    height_base = max(a.bbox.height, b.bbox.height)
    horizontal_gap = max(0.0, max(a.bbox.x1, b.bbox.x1) - min(a.bbox.x2, b.bbox.x2))
    vertical_gap = max(0.0, b.bbox.y1 - a.bbox.y2) if a.page_number == b.page_number else 0.0
    same_column = (
        math.nan
        if a.column_id is None or b.column_id is None
        else float(a.column_id == b.column_id)
    )
    return {
        "vertical_gap_norm": vertical_gap / max(a.page_height, 1),
        "horizontal_gap_norm": horizontal_gap / max(a.page_width, 1),
        "horizontal_overlap_ratio": _overlap(a.bbox.x1, a.bbox.x2, b.bbox.x1, b.bbox.x2) / width_base,
        "vertical_overlap_ratio": _overlap(a.bbox.y1, a.bbox.y2, b.bbox.y1, b.bbox.y2) / height_base,
        "left_alignment_delta_norm": abs(a.bbox.x1 - b.bbox.x1) / max(a.page_width, 1),
        "right_alignment_delta_norm": abs(a.bbox.x2 - b.bbox.x2) / max(a.page_width, 1),
        "indentation_delta_norm": (b.bbox.x1 - a.bbox.x1) / max(a.page_width, 1),
        "width_ratio": min(a.bbox.width, b.bbox.width) / width_base,
        "height_ratio": min(a.bbox.height, b.bbox.height) / height_base,
        "same_column": same_column,
        "same_page": float(a.page_number == b.page_number),
        "page_distance": float(abs(b.page_number - a.page_number)),
    }
