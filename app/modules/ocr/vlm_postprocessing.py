"""Canonical cleanup helpers for VLM OCR text.

Raw model text remains untouched for diagnostics/provenance. These helpers only clean
canonical text that is returned by the OCR service and later persisted to plain-text
artifacts.
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.modules.ocr.text_normalization import normalize_persian_ocr_text
from app.modules.ocr.types import OCRLine


_ROLE_TOKEN_RE = re.compile(
    r"</?\|(?:assistant|user|system)\|?>|<\|(?:assistant|user|system)\|>",
    re.IGNORECASE,
)
_BLOCK_TOKEN_RE = re.compile(
    r"\\?\s*\|im_begin_block\|>",
    re.IGNORECASE,
)
_ESCAPED_LEADING_MARKER_RE = re.compile(r"(?m)^\s*\\\*\s*")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_vlm_canonical_text(value: str) -> str:
    """Remove residual chat/template artifacts without rewriting OCR content."""

    cleaned = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _ROLE_TOKEN_RE.sub("", cleaned)
    cleaned = _BLOCK_TOKEN_RE.sub("", cleaned)
    # DeepSeek sometimes escapes a Markdown bullet directly before Persian text.
    # It is presentation noise, not source-document content.
    cleaned = _ESCAPED_LEADING_MARKER_RE.sub("", cleaned)
    cleaned = _EXCESS_BLANK_LINES_RE.sub("\n\n", cleaned)
    return normalize_persian_ocr_text(cleaned)


def clean_vlm_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Clean canonical line text while preserving ``raw_text`` byte-for-byte."""

    return [replace(line, text=clean_vlm_canonical_text(line.text)) for line in lines]
