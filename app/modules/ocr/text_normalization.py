"""Conservative OCR text normalization for canonical downstream text.

Raw model text must remain untouched for traceability. These helpers operate only on
canonical/logical text that is shown to users and written to plain-text artifacts.
"""

from __future__ import annotations

import re
import unicodedata


_BIDI_CONTROLS_RE = re.compile("[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_HORIZONTAL_WS_RE = re.compile(r"[^\S\r\n]+")
_ZWNJ_SPACING_RE = re.compile(r"[ \t]*\u200c[ \t]*")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([،؛؟!?.,:])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{«])[ \t]+")
_SPACE_BEFORE_CLOSE_RE = re.compile(r"[ \t]+([\)\]\}»])")

_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "٠": "۰",
        "١": "۱",
        "٢": "۲",
        "٣": "۳",
        "٤": "۴",
        "٥": "۵",
        "٦": "۶",
        "٧": "۷",
        "٨": "۸",
        "٩": "۹",
    }
)


def normalize_ocr_text(value: str) -> str:
    """Normalize logical OCR text without changing lexical content.

    This removes transport/display bidi controls, canonicalizes Unicode composition,
    normalizes horizontal whitespace and conservative punctuation spacing, and keeps
    ZWNJ intact. It deliberately does not spell-correct or infer missing characters.
    """

    normalized = unicodedata.normalize("NFC", value or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _BIDI_CONTROLS_RE.sub("", normalized)

    lines: list[str] = []
    for line in normalized.split("\n"):
        line = _HORIZONTAL_WS_RE.sub(" ", line).strip()
        line = _ZWNJ_SPACING_RE.sub("\u200c", line)
        line = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)
        line = _SPACE_AFTER_OPEN_RE.sub(r"\1", line)
        line = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", line)
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_persian_ocr_text(value: str) -> str:
    """Normalize canonical Persian OCR text while preserving raw model output separately."""

    return normalize_ocr_text((value or "").translate(_ARABIC_TO_PERSIAN))
