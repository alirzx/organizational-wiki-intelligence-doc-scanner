from __future__ import annotations

import math
from collections.abc import Sequence

from app.text_processing.linguistic_features import extract_linguistic_features
from app.text_processing.structural_features import extract_structural_features
from app.text_processing.types import CandidatePair, FeatureDescriptor, FeatureSchema, FeatureVector, TextBlock
from app.text_processing.visual_features import extract_visual_features


_BASE_NAMES = tuple(
    list(extract_visual_features.__annotations__)[:0]
)


def feature_schema() -> FeatureSchema:
    sample_names = (
        "vertical_gap_norm", "horizontal_gap_norm", "horizontal_overlap_ratio", "vertical_overlap_ratio",
        "left_alignment_delta_norm", "right_alignment_delta_norm", "indentation_delta_norm", "width_ratio",
        "height_ratio", "same_column", "same_page", "page_distance", "a_list_start", "b_list_start",
        "a_numeric_start", "b_numeric_start", "a_heading_like", "b_heading_like", "indent_delta_norm",
        "line_height_ratio", "line_width_ratio", "a_sentence_complete", "b_sentence_complete",
        "b_continuation_start", "a_list_start_linguistic", "b_list_start_linguistic", "a_token_count",
        "b_token_count", "length_ratio", "reason_reading_window", "reason_same_column", "reason_spatial_near",
        "reason_structural_match", "reason_page_boundary", "semantic_similarity", "semantic_available",
    )
    return FeatureSchema(
        version="grouping.features.v1",
        normalizer_version="normalizer.v1",
        features=tuple(
            FeatureDescriptor(
                name,
                missing_policy="native_missing" if name in {"same_column", "semantic_similarity"} else "forbidden",
            )
            for name in sample_names
        ),
    )


def extract_features(
    a: TextBlock,
    b: TextBlock,
    candidate: CandidatePair,
    *,
    semantic_similarity: float | None = None,
) -> FeatureVector:
    values = extract_visual_features(a, b) | extract_structural_features(a, b) | extract_linguistic_features(a, b)
    reason_values = {f"reason_{reason.value}": 1.0 for reason in candidate.reasons}
    values.update(reason_values)
    values["semantic_similarity"] = math.nan if semantic_similarity is None else semantic_similarity
    values["semantic_available"] = float(semantic_similarity is not None)
    schema = feature_schema()
    vector = FeatureVector(candidate.pair_id, schema.sha256, tuple(float(values.get(name, 0.0)) for name in schema.names))
    vector.validate(schema)
    return vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must have equal non-zero dimensions")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else max(-1.0, min(1.0, dot / (norm_a * norm_b)))
