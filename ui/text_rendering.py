from __future__ import annotations

from html import escape
from typing import Mapping, Sequence

from app.modules.ocr.text_normalization import normalize_ocr_text


def canonical_ocr_display_text(paragraphs: Sequence[Mapping]) -> str:
    """Return human-facing logical OCR text; raw model order is debug-only."""

    values: list[str] = []
    for paragraph in paragraphs:
        value = paragraph.get("text") or paragraph.get("raw_text") or ""
        normalized = normalize_ocr_text(str(value))
        if normalized:
            values.append(normalized)
    return "\n\n".join(values)


def bidi_safe_text_html(value: str) -> str:
    """Render logical Unicode text without mutating it or applying visual reversal."""

    safe = escape(value).replace("\n", "<br>")
    return (
        '<div dir="auto" lang="fa" style="unicode-bidi: plaintext; '
        'white-space: normal; text-align: start; line-height: 1.9; '
        'font-size: 1rem; padding: .75rem 1rem; border: 1px solid '
        'rgba(128,128,128,.25); border-radius: .5rem;">'
        f"{safe}</div>"
    )
