from __future__ import annotations

import re

from app.text_processing.types import TextBlock


_DIGITS = r"0-9۰-۹٠-٩"
_BULLETS = r"•●▪◦‣⁃*\-–—"
_PERSIAN_ALPHA = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
_PERSIAN_MARKER = r"(?:الف|ب|پ|ت|ث|ج|چ|ح|خ|د|ذ|ر|ز|ژ|س|ش|ص|ض|ط|ظ|ع|غ|ف|ق|ک|گ|ل|م|ن|و|ه|ی)"
_LIST_START = re.compile(
    rf"^\s*(?:(?:[{_DIGITS}]+|[A-Za-z]|{_PERSIAN_MARKER})\s*[.)\-:ـ]|[{_BULLETS}]|(?:ماده|تبصره|بند)\s*[{_DIGITS}]+\s*[-.:)]?)\s*",
    re.IGNORECASE,
)
_NUMERIC_START = re.compile(rf"^\s*[{_DIGITS}]+\s*[.)\-:ـ]")


def is_list_start(text: str) -> bool:
    return bool(_LIST_START.match(text))


def is_heading_like(text: str) -> bool:
    value = text.strip()
    return bool(value) and len(value.split()) <= 12 and not value.endswith((".", "؟", "?", ";", "؛", ",", "،"))


def extract_structural_features(a: TextBlock, b: TextBlock) -> dict[str, float]:
    height = max(a.bbox.height, b.bbox.height)
    return {
        "a_list_start": float(is_list_start(a.normalized_text)),
        "b_list_start": float(is_list_start(b.normalized_text)),
        "a_numeric_start": float(bool(_NUMERIC_START.match(a.normalized_text))),
        "b_numeric_start": float(bool(_NUMERIC_START.match(b.normalized_text))),
        "a_heading_like": float(is_heading_like(a.normalized_text)),
        "b_heading_like": float(is_heading_like(b.normalized_text)),
        "indent_delta_norm": abs(b.bbox.x1 - a.bbox.x1) / max(a.page_width, 1),
        "line_height_ratio": min(a.bbox.height, b.bbox.height) / height,
        "line_width_ratio": min(a.bbox.width, b.bbox.width) / max(a.bbox.width, b.bbox.width),
    }
