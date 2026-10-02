from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict

from app.text_processing.types import CandidatePair, RelationshipDecision, RelationshipPrediction, RelationshipLabel, TextBlock


@dataclass(frozen=True)
class GraphResolution:
    components: tuple[tuple[str, ...], ...]
    accepted: tuple[RelationshipPrediction, ...]
    uncertain: tuple[RelationshipPrediction, ...]
    rejected: tuple[RelationshipPrediction, ...]


def resolve_components(
    blocks: list[TextBlock],
    predictions: list[RelationshipPrediction],
    *,
    max_component_size: int = 100,
    candidates: list[CandidatePair] | None = None,
) -> GraphResolution:
    ordered = sorted(blocks, key=lambda block: block.document_order)
    by_id = {block.block_id: block for block in ordered}
    if len(by_id) != len(blocks) or len({b.document_id for b in blocks}) > 1:
        raise ValueError('graph requires unique blocks from one document')
    parent = {block.block_id: block.block_id for block in ordered}
    members = {block.block_id: {block.block_id} for block in ordered}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    accepted: list[RelationshipPrediction] = []
    uncertain = [p for p in predictions if p.decision == RelationshipDecision.UNCERTAIN]
    rejected: list[RelationshipPrediction] = [p for p in predictions if p.decision == RelationshipDecision.SEPARATE]
    ranked = sorted(
        (p for p in predictions if p.decision == RelationshipDecision.MERGE),
        key=lambda p: (-p.probabilities[RelationshipLabel.SAME_GROUP], p.pair_id),
    )
    if not ranked:
        return GraphResolution(tuple((b.block_id,) for b in ordered), (), tuple(uncertain), tuple(rejected))
    pair_by_id = ({p.pair_id:(p.block_a_id,p.block_b_id) for p in candidates} if candidates is not None
                  else {p.pair_id:(p.block_a_id,p.block_b_id) for p in predictions
                        if p.block_a_id is not None and p.block_b_id is not None})
    forbidden = {b.block_id:set() for b in ordered}
    positions = {b.block_id:i for i,b in enumerate(ordered)}
    heading_prefix = [0]
    column_positions = {}
    column_prefix = defaultdict(lambda:[0])
    for block in ordered:
        heading_prefix.append(heading_prefix[-1]+int(block.block_type == 'heading'))
        if block.column_id is not None:
            prefix = column_prefix[block.column_id]
            column_positions[block.block_id] = len(prefix)-1
            prefix.append(prefix[-1]+int(block.block_type == 'heading'))
    for prediction in rejected:
        endpoints = pair_by_id.get(prediction.pair_id)
        if endpoints and all(item in by_id for item in endpoints):
            left,right = endpoints
            forbidden[left].add(right)
            forbidden[right].add(left)
    for prediction in ranked:
        endpoints = pair_by_id.get(prediction.pair_id)
        if not endpoints or any(item not in by_id for item in endpoints):
            rejected.append(replace(prediction, decision=RelationshipDecision.GUARD_REJECTED, guard_reasons=("unknown_pair",)))
            continue
        left, right = endpoints
        a,b = by_id[left],by_id[right]
        edge_reasons = []
        prefix,left_position,right_position = heading_prefix,positions[left],positions[right]
        if a.column_id is not None and a.column_id == b.column_id:
            prefix = column_prefix[a.column_id]
            left_position,right_position = column_positions[left],column_positions[right]
        if positions[right] <= positions[left]:
            edge_reasons.append('invalid_order')
        elif prefix[right_position+1] > prefix[left_position+1]:
            edge_reasons.append('heading_boundary')
        if b.page_number != a.page_number and (b.page_number != a.page_number+1
            or a.bbox.y2 < .8*a.page_height or b.bbox.y1 > .2*b.page_height):
            edge_reasons.append('cross_page_boundary')
        if edge_reasons:
            rejected.append(replace(prediction,decision=RelationshipDecision.GUARD_REJECTED,guard_reasons=tuple(edge_reasons)))
            continue
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            accepted.append(prediction)
            continue
        combined = members[root_left] | members[root_right]
        columns = {by_id[item].column_id for item in combined if by_id[item].column_id is not None}
        reasons: list[str] = []
        if len(combined) > max_component_size:
            reasons.append("max_component_size")
        if len(columns) > 1:
            reasons.append("incompatible_columns")
        if any(forbidden[item] & combined for item in combined):
            reasons.append('explicit_separation')
        if reasons:
            rejected.append(replace(prediction, decision=RelationshipDecision.GUARD_REJECTED, guard_reasons=tuple(reasons)))
            continue
        if len(members[root_left]) < len(members[root_right]):
            root_left,root_right = root_right,root_left
        parent[root_right] = root_left
        members[root_left] = combined
        del members[root_right]
        accepted.append(prediction)
    components: dict[str, list[str]] = {}
    for block in ordered:
        components.setdefault(find(block.block_id), []).append(block.block_id)
    return GraphResolution(tuple(tuple(value) for value in components.values()), tuple(accepted), tuple(uncertain), tuple(rejected))
