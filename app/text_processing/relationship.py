from __future__ import annotations

from app.text_processing.types import CandidatePair, RelationshipDecision, RelationshipLabel, RelationshipPrediction


def prediction_from_score(
    candidate: CandidatePair,
    score: float,
    *,
    merge_threshold: float = 0.75,
    uncertain_lower: float = 0.55,
    semantic_available: bool = False,
    model_package_id: str = "injected",
) -> RelationshipPrediction:
    score = min(1.0, max(0.0, score))
    decision = (
        RelationshipDecision.MERGE if score >= merge_threshold
        else RelationshipDecision.UNCERTAIN if score >= uncertain_lower
        else RelationshipDecision.SEPARATE
    )
    label = RelationshipLabel.SAME_GROUP if score >= 0.5 else RelationshipLabel.NEW_GROUP
    probabilities = {RelationshipLabel.SAME_GROUP: score, RelationshipLabel.NEW_GROUP: 1 - score}
    return RelationshipPrediction(
        pair_id=candidate.pair_id,
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
        decision=decision,
        semantic_available=semantic_available,
        model_package_id=model_package_id,
    )
