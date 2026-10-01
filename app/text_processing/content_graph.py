from __future__ import annotations

from dataclasses import dataclass, replace

from app.text_processing.types import RelationshipDecision, RelationshipPrediction, TextBlock


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
) -> GraphResolution:
    ordered = sorted(blocks, key=lambda block: block.document_order)
    by_id = {block.block_id: block for block in ordered}
    parent = {block.block_id: block.block_id for block in ordered}
    members = {block.block_id: {block.block_id} for block in ordered}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    pair_by_id: dict[str, tuple[str, str]] = {}
    # Pair IDs are opaque, so recover endpoints by regenerating against the small block set.
    from app.text_processing.ids import stable_pair_id
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            for version in ("candidate.v1", "cross-page.v1"):
                pair_by_id[stable_pair_id(a.block_id, b.block_id, version)] = (a.block_id, b.block_id)

    accepted: list[RelationshipPrediction] = []
    uncertain = [p for p in predictions if p.decision == RelationshipDecision.UNCERTAIN]
    rejected: list[RelationshipPrediction] = [p for p in predictions if p.decision == RelationshipDecision.SEPARATE]
    ranked = sorted(
        (p for p in predictions if p.decision == RelationshipDecision.MERGE),
        key=lambda p: (-p.probabilities.get(next(iter(p.probabilities)), 0.0), p.pair_id),
    )
    for prediction in ranked:
        endpoints = pair_by_id.get(prediction.pair_id)
        if not endpoints:
            rejected.append(replace(prediction, decision=RelationshipDecision.GUARD_REJECTED, guard_reasons=("unknown_pair",)))
            continue
        left, right = endpoints
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
        if reasons:
            rejected.append(replace(prediction, decision=RelationshipDecision.GUARD_REJECTED, guard_reasons=tuple(reasons)))
            continue
        parent[root_right] = root_left
        members[root_left] = combined
        accepted.append(prediction)
    components: dict[str, list[str]] = {}
    for block in ordered:
        components.setdefault(find(block.block_id), []).append(block.block_id)
    return GraphResolution(tuple(tuple(value) for value in components.values()), tuple(accepted), tuple(uncertain), tuple(rejected))
