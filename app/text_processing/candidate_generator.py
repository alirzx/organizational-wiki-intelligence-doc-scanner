from __future__ import annotations

from dataclasses import dataclass

from app.text_processing.ids import stable_pair_id
from app.text_processing.structural_features import is_list_start
from app.text_processing.types import CandidatePair, CandidateReason, TextBlock


@dataclass(frozen=True)
class CandidateConfig:
    reading_lookahead: int = 3
    spatial_gap_ratio: float = 0.08
    max_pairs: int = 10_000
    cross_page_window: int = 3
    max_page_distance: int = 1
    schema_version: str = "candidate.v1"


def generate_candidates(blocks: list[TextBlock], config: CandidateConfig | None = None) -> list[CandidatePair]:
    config = config or CandidateConfig()
    ordered = sorted(blocks, key=lambda item: item.document_order)
    reasons_by_pair: dict[tuple[str, str], set[CandidateReason]] = {}
    lookup = {block.block_id: block for block in ordered}
    for index, a in enumerate(ordered):
        for b in ordered[index + 1:]:
            if b.page_number != a.page_number:
                break
            reasons: set[CandidateReason] = set()
            if b.content_ordinal - a.content_ordinal <= config.reading_lookahead:
                reasons.add(CandidateReason.READING_WINDOW)
            if a.column_id is not None and a.column_id == b.column_id:
                reasons.add(CandidateReason.SAME_COLUMN)
            if max(0.0, b.bbox.y1 - a.bbox.y2) / a.page_height <= config.spatial_gap_ratio:
                reasons.add(CandidateReason.SPATIAL_NEAR)
            if is_list_start(a.normalized_text) or is_list_start(b.normalized_text):
                reasons.add(CandidateReason.STRUCTURAL_MATCH)
            if reasons:
                reasons_by_pair.setdefault((a.block_id, b.block_id), set()).update(reasons)
    pages: dict[int, list[TextBlock]] = {}
    for block in ordered:
        pages.setdefault(block.page_number, []).append(block)
    for page_number in sorted(pages):
        for next_page in range(page_number + 1, page_number + config.max_page_distance + 1):
            if next_page not in pages:
                continue
            tails = pages[page_number][-config.cross_page_window:]
            heads = pages[next_page][:config.cross_page_window]
            for a in tails:
                for b in heads:
                    same_column = a.column_id is None or b.column_id is None or a.column_id == b.column_id
                    indent_close = abs(a.bbox.x1 - b.bbox.x1) / max(a.page_width, 1) <= 0.08
                    structural = is_list_start(a.normalized_text) == is_list_start(b.normalized_text)
                    if same_column and indent_close and structural:
                        reasons_by_pair.setdefault((a.block_id, b.block_id), set()).add(CandidateReason.PAGE_BOUNDARY)
    pairs = [
        CandidatePair(
            pair_id=stable_pair_id(a_id, b_id, config.schema_version),
            block_a_id=a_id,
            block_b_id=b_id,
            cross_page=lookup[a_id].page_number != lookup[b_id].page_number,
            page_distance=lookup[b_id].page_number - lookup[a_id].page_number,
            reasons=tuple(reasons),
        )
        for (a_id, b_id), reasons in reasons_by_pair.items()
    ]
    pairs.sort(key=lambda pair: (lookup[pair.block_a_id].document_order, lookup[pair.block_b_id].document_order))
    return pairs[: max(0, config.max_pairs)]
