from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict

from app.text_processing.ids import stable_pair_id
from app.text_processing.structural_features import is_list_start, is_heading_like
from app.text_processing.types import CandidatePair, CandidateReason, TextBlock


@dataclass(frozen=True)
class CandidateConfig:
    reading_lookahead: int = 3
    spatial_gap_ratio: float = 0.08
    max_pairs: int = 10_000
    cross_page_window: int = 3
    max_page_distance: int = 1
    schema_version: str = "candidate.v1"
    optional_pairs_per_block: int = 25
    adjacent_only: bool = False


@dataclass(frozen=True)
class CandidateGeneration:
    pairs: list[CandidatePair]
    diagnostics: dict


def generate_candidates(blocks: list[TextBlock], config: CandidateConfig | None = None) -> list[CandidatePair]:
    return generate_candidate_batch(blocks, config).pairs


def generate_candidate_batch(blocks: list[TextBlock], config: CandidateConfig | None = None) -> CandidateGeneration:
    config = config or CandidateConfig()
    if config.reading_lookahead < 1 or config.cross_page_window < 1 or config.max_page_distance < 0:
        raise ValueError('candidate windows must be positive and page distance nonnegative')
    if len({b.block_id for b in blocks}) != len(blocks) or len({b.document_id for b in blocks}) > 1:
        raise ValueError('candidates require unique block IDs from one document')
    ordered = sorted(blocks, key=lambda item: item.document_order)
    lookup = {block.block_id: block for block in ordered}
    flags = {b.block_id: (is_list_start(b.normalized_text), is_heading_like(b.normalized_text)) for b in ordered}
    mandatory = {}
    adjacent_targets = set()
    boundary_targets = set()
    optional = defaultdict(list)
    pages = defaultdict(list)
    for block in ordered:
        pages[block.page_number].append(block)
    def eligible(a, b):
        return a.column_id is None or b.column_id is None or a.column_id == b.column_id
    for page, items in pages.items():
        for i, a in enumerate(items):
            for j in range(i+1, min(len(items), i+config.reading_lookahead+1)):
                b = items[j]
                if not eligible(a,b):
                    continue
                adjacent = b.content_ordinal == a.content_ordinal+1
                if adjacent:
                    adjacent_targets.add((a.block_id,b.block_id))
                if config.adjacent_only and not adjacent:
                    continue
                reasons = {CandidateReason.READING_WINDOW}
                if adjacent:
                    reasons.add(CandidateReason.ADJACENT_SAME_PAGE)
                gap = max(0., b.bbox.y1-a.bbox.y2)/a.page_height
                near = gap <= config.spatial_gap_ratio
                same_column = a.column_id is not None and a.column_id == b.column_id
                if same_column and near:
                    reasons.update((CandidateReason.SAME_COLUMN, CandidateReason.SAME_COLUMN_NEARBY))
                if near:
                    reasons.add(CandidateReason.SPATIAL_NEAR)
                if min(a.bbox.x2,b.bbox.x2) > max(a.bbox.x1,b.bbox.x1):
                    reasons.add(CandidateReason.GEOMETRY_OVERLAP)
                structural = flags[a.block_id][0] or flags[b.block_id][0]
                if structural and near:
                    reasons.update((CandidateReason.STRUCTURAL_MATCH, CandidateReason.LIST_CONTINUATION))
                heading = flags[a.block_id][1] and not flags[b.block_id][1] and near
                if heading:
                    reasons.add(CandidateReason.HEADING_TO_BODY)
                key = (a.block_id,b.block_id)
                if adjacent or (same_column and near) or (structural and near) or heading:
                    mandatory[key] = reasons
                else:
                    optional[page].append((key,reasons))
        # Local column successors survive interleaved reading order without a global scan.
        columns = defaultdict(list)
        for block in items:
            if block.column_id is not None:
                columns[block.column_id].append(block)
        for column in columns.values():
            for a,b in zip(column,column[1:]):
                if max(0.,b.bbox.y1-a.bbox.y2)/a.page_height <= config.spatial_gap_ratio:
                    mandatory.setdefault((a.block_id,b.block_id),set()).update((CandidateReason.SAME_COLUMN, CandidateReason.SAME_COLUMN_NEARBY))
    for page_number in sorted(pages):
        for next_page in range(page_number + 1, page_number + config.max_page_distance + 1):
            if next_page not in pages:
                continue
            tails = pages[page_number][-config.cross_page_window:]
            heads = pages[next_page][:config.cross_page_window]
            for a in tails:
                for b in heads:
                    indent_close = abs(a.bbox.x1 - b.bbox.x1) / max(a.page_width, 1) <= 0.08
                    structural = flags[a.block_id][0] == flags[b.block_id][0]
                    boundary = a.bbox.y2 >= .8*a.page_height and b.bbox.y1 <= .2*b.page_height
                    if config.adjacent_only and (a != pages[page_number][-1] or b != pages[next_page][0]):
                        continue
                    if eligible(a,b) and indent_close and structural and boundary:
                        boundary_targets.add((a.block_id,b.block_id))
                        mandatory.setdefault((a.block_id, b.block_id), set()).update((CandidateReason.PAGE_BOUNDARY, CandidateReason.ADJACENT_CROSS_PAGE))
    # Equal page allotments plus per-source quotas; caps apply only to optional work.
    reasons_by_pair = dict(mandatory)
    budget = max(0, config.max_pairs)
    page_quota = budget//max(1,len(pages))
    remainder = budget % max(1,len(pages))
    dropped = 0
    source_counts = Counter()
    optional_count = 0
    for page_index,page in enumerate(sorted(pages)):
        accepted_page = 0
        for key,reasons in optional[page]:
            if key in reasons_by_pair:
                reasons_by_pair[key].update(reasons)
                continue
            if accepted_page >= page_quota+(page_index<remainder) or source_counts[key[0]] >= config.optional_pairs_per_block:
                dropped += 1
                continue
            reasons_by_pair[key] = reasons
            source_counts[key[0]] += 1
            accepted_page += 1
            optional_count += 1
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
    reason_counts = Counter(r.value for p in pairs for r in p.reasons)
    actual = set(reasons_by_pair)
    return CandidateGeneration(pairs, {'total':len(pairs), 'mandatory':len(mandatory),
        'optional':optional_count, 'truncated':dropped, 'per_reason':dict(reason_counts),
        'per_page':dict(Counter(str(lookup[p.block_a_id].page_number) for p in pairs)),
        'adjacent_candidate_recall':len(actual & adjacent_targets)/len(adjacent_targets) if adjacent_targets else None,
        'boundary_candidate_recall':len(actual & boundary_targets)/len(boundary_targets) if boundary_targets else None})
