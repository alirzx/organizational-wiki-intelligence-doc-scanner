"""Geometry- and structure-aware grouping of OCR text lines into V1 paragraph objects.

Stage 1 must preserve what the OCR model actually saw while still producing plain text
that downstream stages can read. ``raw_text`` therefore remains untouched, while
``text`` may apply conservative layout normalization such as list-marker spacing.

This module deliberately avoids semantic section classification; that belongs to later
Wiki Hami stages.
"""

from __future__ import annotations

import re
from statistics import median

from app.modules.ocr.text_normalization import normalize_ocr_text
from app.modules.ocr.types import OCRLine, OCRParagraph
from app.schemas.common import BBox, Point, Polygon


# Full Persian alphabetic order used by enumerated legal/administrative lists.
# Arabic-script glyph variants are canonicalized before sequence checks.
_PERSIAN_LIST_SEQUENCE = (
    "الف",
    "ب",
    "پ",
    "ت",
    "ث",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ک",
    "گ",
    "ل",
    "م",
    "ن",
    "و",
    "ه",
    "ی",
)
_ENGLISH_LIST_SEQUENCE = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))

_PERSIAN_MARKER_ALIASES = (
    "الف",
    "هـ",
    "ب",
    "پ",
    "ت",
    "ث",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ک",
    "ك",
    "گ",
    "ل",
    "م",
    "ن",
    "و",
    "ه",
    "ی",
    "ي",
    "ى",
)
_PERSIAN_MARKER_CANONICAL = {"هـ": "ه", "ك": "ک", "ي": "ی", "ى": "ی"}
_MARKER_SEPARATORS = "-–—ـ.:؛،)"

_STRUCTURAL_START_RE = re.compile(
    r"^(?:ماده|تبصره|فصل|بخش|بند|chapter|section|article|clause)"
    r"(?=$|\s|[-–—ـ.:؛،()0-9۰-۹٠-٩])",
    re.IGNORECASE,
)
_NUMERIC_LIST_RE = re.compile(
    r"^\s*[\(\[]?[0-9۰-۹٠-٩]+(?:[\)\]]|[-–—ـ.:])\s*"
)
_SYMBOL_BULLET_RE = re.compile(r"^\s*(?:[•●▪◦◾■□*]|[-–—])\s+")


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    denom = max(1.0, min(a.width, b.width))
    return overlap / denom


def _union_bbox(lines: list[OCRLine]) -> BBox:
    return BBox(
        x1=min(line.bbox.x1 for line in lines),
        y1=min(line.bbox.y1 for line in lines),
        x2=max(line.bbox.x2 for line in lines),
        y2=max(line.bbox.y2 for line in lines),
    )


def _bbox_polygon(bbox: BBox) -> Polygon:
    return Polygon(
        points=[
            Point(x=bbox.x1, y=bbox.y1),
            Point(x=bbox.x2, y=bbox.y1),
            Point(x=bbox.x2, y=bbox.y2),
            Point(x=bbox.x1, y=bbox.y2),
        ]
    )


def _normalize_inline_whitespace(value: str) -> str:
    """Apply canonical text cleanup while preserving language and lexical content."""

    return normalize_ocr_text(value)


def _canonical_persian_marker(value: str) -> str:
    return _PERSIAN_MARKER_CANONICAL.get(value, value)


def _sequence_position(kind: str, marker: str) -> int | None:
    sequence = _PERSIAN_LIST_SEQUENCE if kind == "fa" else _ENGLISH_LIST_SEQUENCE
    try:
        return sequence.index(marker)
    except ValueError:
        return None


def _markers_are_sequential(
    previous: tuple[str, str, str, str],
    current: tuple[str, str, str, str],
) -> bool:
    previous_kind, previous_marker, _, _ = previous
    current_kind, current_marker, _, _ = current
    if previous_kind != current_kind:
        return False
    previous_position = _sequence_position(previous_kind, previous_marker)
    current_position = _sequence_position(current_kind, current_marker)
    return (
        previous_position is not None
        and current_position is not None
        and current_position == previous_position + 1
    )


def _persian_marker_candidate(text: str) -> tuple[str, str, str, str] | None:
    value = _normalize_inline_whitespace(text)
    # Longest aliases first so ``الف`` is checked before single-letter candidates.
    for alias in sorted(_PERSIAN_MARKER_ALIASES, key=len, reverse=True):
        if not value.startswith(alias):
            continue

        tail = value[len(alias) :]
        if not tail:
            continue

        marker = _canonical_persian_marker(alias)
        if tail[0].isspace():
            body = tail.lstrip()
            if body and body[0] in _MARKER_SEPARATORS:
                body = body.lstrip(_MARKER_SEPARATORS + " ").strip()
                return "fa", marker, body, "strong"
            return "fa", marker, body.strip(), "weak"

        if tail[0] in _MARKER_SEPARATORS:
            body = tail.lstrip(_MARKER_SEPARATORS + " ").strip()
            return "fa", marker, body, "strong"

        # OCR occasionally glues a Persian list marker to its first word. This is only
        # trusted later when neighboring markers prove that a list sequence exists.
        return "fa", marker, tail.strip(), "glued"

    return None


def _english_marker_candidate(text: str) -> tuple[str, str, str, str] | None:
    value = _normalize_inline_whitespace(text)
    match = re.match(r"^([A-Za-z])(?:(\s*[-–—.:)])|(\s*\)))?\s*(.*)$", value)
    if not match:
        return None

    marker = match.group(1).upper()
    separator = match.group(2) or match.group(3)
    body = (match.group(4) or "").strip()
    if separator:
        return "en", marker, body, "strong"

    # A bare ``A text`` form is intentionally weak because ordinary English sentences
    # often start with the article "A". It is accepted only when a neighboring sequence
    # such as A/B/C confirms it.
    if len(value) > 1 and value[1].isspace():
        return "en", marker, body, "weak"
    return None


def _marker_candidate(text: str) -> tuple[str, str, str, str] | None:
    return _persian_marker_candidate(text) or _english_marker_candidate(text)


def _accepted_list_markers(
    ordered: list[OCRLine],
) -> dict[int, tuple[str, str, str, str]]:
    """Conservatively identify alphabetic list markers across OCR lines.

    Strong markers with visible punctuation are authoritative. Weak/glued markers are
    accepted only when neighboring lines establish a real Persian or English alphabetic
    sequence. This keeps normal prose such as ``و در صورت...`` or ``A service...`` from
    being split into fake list items.
    """

    candidates = [_marker_candidate(line.text) for line in ordered]
    accepted: set[int] = {
        index
        for index, candidate in enumerate(candidates)
        if candidate is not None and candidate[3] == "strong"
    }

    candidate_indexes = [index for index, candidate in enumerate(candidates) if candidate is not None]
    runs: list[list[int]] = []
    current: list[int] = []

    for index in candidate_indexes:
        if not current:
            current = [index]
            continue

        previous_index = current[-1]
        previous = candidates[previous_index]
        candidate = candidates[index]
        assert previous is not None and candidate is not None

        # Permit one visual wrapped line between two list-item starts.
        close_enough = index - previous_index <= 2
        if close_enough and _markers_are_sequential(previous, candidate):
            current.append(index)
        else:
            runs.append(current)
            current = [index]

    if current:
        runs.append(current)

    for run in runs:
        strengths = [candidates[index][3] for index in run if candidates[index] is not None]
        if len(run) >= 3 or (len(run) >= 2 and "strong" in strengths):
            accepted.update(run)

    return {
        index: candidates[index]
        for index in sorted(accepted)
        if candidates[index] is not None
    }


def _is_hard_block_start(
    text: str,
    marker: tuple[str, str, str, str] | None,
) -> bool:
    value = _normalize_inline_whitespace(text)
    return bool(
        marker is not None
        or _STRUCTURAL_START_RE.match(value)
        or _NUMERIC_LIST_RE.match(value)
        or _SYMBOL_BULLET_RE.match(value)
    )


def _normalized_line(
    text: str,
    marker: tuple[str, str, str, str] | None,
) -> str:
    value = _normalize_inline_whitespace(text)
    if marker is None:
        return value

    _, marker_name, body, _ = marker
    body = _normalize_inline_whitespace(body)
    if not body:
        return marker_name
    return f"{marker_name} - {body}"


def group_lines_into_paragraphs(
    lines: list[OCRLine],
    *,
    max_gap_ratio: float = 1.8,
    min_x_overlap: float = 0.15,
) -> list[OCRParagraph]:
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (line.bbox.y1, line.bbox.x1))
    typical_height = max(1.0, median(max(1.0, line.bbox.height) for line in ordered))
    max_gap = typical_height * max_gap_ratio
    accepted_markers = _accepted_list_markers(ordered)
    index_by_identity = {id(line): index for index, line in enumerate(ordered)}

    groups: list[list[OCRLine]] = []
    current: list[OCRLine] = [ordered[0]]

    for line in ordered[1:]:
        line_index = index_by_identity[id(line)]
        marker = accepted_markers.get(line_index)
        previous = current[-1]
        vertical_gap = line.bbox.y1 - previous.bbox.y2
        same_column = _x_overlap_ratio(_union_bbox(current), line.bbox) >= min_x_overlap
        hard_block_start = _is_hard_block_start(line.text, marker)

        # Every detected list item/section start begins a separate paragraph. Wrapped
        # visual lines without a structural marker stay inside the current paragraph and
        # retain an explicit newline in ``text``/``raw_text``.
        if (
            not hard_block_start
            and vertical_gap <= max_gap
            and vertical_gap >= -typical_height
            and same_column
        ):
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)

    paragraphs: list[OCRParagraph] = []
    for group in groups:
        bbox = _union_bbox(group)
        raw_text = "\n".join(line.raw_text if line.raw_text is not None else line.text for line in group)

        normalized_lines: list[str] = []
        for line in group:
            line_index = index_by_identity[id(line)]
            normalized = _normalized_line(line.text, accepted_markers.get(line_index))
            if normalized:
                normalized_lines.append(normalized)
        normalized = "\n".join(normalized_lines)

        weighted_area = [max(1.0, line.bbox.width * line.bbox.height) for line in group]
        confidence = sum(
            line.confidence * area
            for line, area in zip(group, weighted_area, strict=True)
        ) / sum(weighted_area)
        paragraphs.append(
            OCRParagraph(
                text=normalized,
                raw_text=raw_text,
                confidence=max(0.0, min(1.0, confidence)),
                bbox=bbox,
                polygon=_bbox_polygon(bbox),
                lines=tuple(group),
            )
        )
    return paragraphs
