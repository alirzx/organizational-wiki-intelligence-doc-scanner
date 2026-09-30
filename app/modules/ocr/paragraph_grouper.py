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

from app.modules.ocr.types import OCRLine, OCRParagraph
from app.schemas.common import BBox, Point, Polygon


# Common legal/administrative Persian enumerator sequence seen in regulations.
# We only accept weak or glued markers when neighboring lines support a sequence,
# which avoids turning an ordinary sentence beginning with "و ..." into a list item.
_LIST_SEQUENCE = ("الف", "ب", "ج", "د", "ه", "و", "ز", "ح", "ط", "ی", "ک")
_MARKER_ALIASES = ("الف", "هـ", "ب", "ج", "د", "ه", "و", "ز", "ح", "ط", "ی", "ي", "ک", "ك")
_MARKER_CANONICAL = {"هـ": "ه", "ي": "ی", "ك": "ک"}
_MARKER_SEPARATORS = "-–—ـ.:؛،)"
_HORIZONTAL_WS_RE = re.compile(r"[^\S\r\n]+")

_STRUCTURAL_START_RE = re.compile(
    r"^(?:ماده|تبصره|فصل|بخش|بند)"
    r"(?=$|\s|[-–—ـ.:؛،()0-9۰-۹٠-٩])"
)
_NUMERIC_LIST_RE = re.compile(
    r"^\s*[\(\[]?[0-9۰-۹٠-٩]+(?:[\)\]]|[-–—ـ.:])\s*"
)
_SYMBOL_BULLET_RE = re.compile(r"^\s*[•●▪◦◾■□*]\s*")


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
    return _HORIZONTAL_WS_RE.sub(" ", value.strip())


def _canonical_marker(value: str) -> str:
    return _MARKER_CANONICAL.get(value, value)


def _next_marker(value: str) -> str | None:
    try:
        index = _LIST_SEQUENCE.index(value)
    except ValueError:
        return None
    if index + 1 >= len(_LIST_SEQUENCE):
        return None
    return _LIST_SEQUENCE[index + 1]


def _marker_candidate(text: str) -> tuple[str, str, str] | None:
    """Return ``(marker, body, strength)`` for a possible Persian list item.

    ``strong`` means a visible separator such as ``-`` survived OCR.
    ``weak`` means only whitespace separates the marker and body.
    ``glued`` means OCR removed even that whitespace (for example ``حدبیر...``).
    Weak/glued candidates are not trusted until a neighboring marker sequence confirms
    them.
    """

    value = _normalize_inline_whitespace(text)
    for alias in _MARKER_ALIASES:
        if not value.startswith(alias):
            continue

        tail = value[len(alias) :]
        if not tail:
            continue

        marker = _canonical_marker(alias)
        if tail[0].isspace():
            body = tail.lstrip()
            if body and body[0] in _MARKER_SEPARATORS:
                body = body.lstrip(_MARKER_SEPARATORS + " ").strip()
                return marker, body, "strong"
            return marker, body.strip(), "weak"

        if tail[0] in _MARKER_SEPARATORS:
            body = tail.lstrip(_MARKER_SEPARATORS + " ").strip()
            return marker, body, "strong"

        return marker, tail.strip(), "glued"

    return None


def _accepted_list_markers(
    ordered: list[OCRLine],
) -> dict[int, tuple[str, str, str]]:
    """Conservatively identify real enumerator lines.

    Visible punctuation is authoritative. When punctuation/spacing was lost by OCR,
    accept the marker only as part of a short-range alphabetical sequence. This handles
    common OCR outputs such as ``الف وزیر...``, ``ب وزیر...`` and ``حدبیر...`` while
    avoiding broad language-level guessing.
    """

    candidates = [_marker_candidate(line.text) for line in ordered]
    accepted: set[int] = {
        index
        for index, candidate in enumerate(candidates)
        if candidate is not None and candidate[2] == "strong"
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
        current_candidate = candidates[index]
        assert previous is not None and current_candidate is not None

        # Allow one wrapped continuation line between two list markers.
        close_enough = index - previous_index <= 2
        sequential = _next_marker(previous[0]) == current_candidate[0]
        if close_enough and sequential:
            current.append(index)
        else:
            runs.append(current)
            current = [index]

    if current:
        runs.append(current)

    for run in runs:
        strengths = [candidates[index][2] for index in run if candidates[index] is not None]
        # Three sequential weak/glued markers are strong evidence of a list. Two are
        # sufficient when OCR retained punctuation on at least one of them.
        if len(run) >= 3 or (len(run) >= 2 and "strong" in strengths):
            accepted.update(run)

    # A mature accepted sequence may end with one OCR-glued marker. This is the common
    # "ز ... / حدبیر ..." failure mode in Persian legal documents.
    accepted_sorted = sorted(accepted)
    if len(accepted_sorted) >= 2:
        last = accepted_sorted[-1]
        if last + 1 < len(ordered):
            previous = candidates[last]
            candidate = candidates[last + 1]
            if (
                previous is not None
                and candidate is not None
                and _next_marker(previous[0]) == candidate[0]
            ):
                accepted.add(last + 1)

    return {
        index: candidates[index]
        for index in sorted(accepted)
        if candidates[index] is not None
    }


def _is_hard_block_start(text: str, marker: tuple[str, str, str] | None) -> bool:
    value = _normalize_inline_whitespace(text)
    return bool(
        marker is not None
        or _STRUCTURAL_START_RE.match(value)
        or _NUMERIC_LIST_RE.match(value)
        or _SYMBOL_BULLET_RE.match(value)
    )


def _normalized_line(text: str, marker: tuple[str, str, str] | None) -> str:
    value = _normalize_inline_whitespace(text)
    if marker is None:
        return value

    marker_name, body, _ = marker
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

        # A detected legal/list item is a logical block even when visual line spacing is
        # identical to the previous row. Otherwise use the original conservative
        # geometry rule so wrapped sentence lines stay together.
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
        raw_text = "\n".join(line.text for line in group)

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
