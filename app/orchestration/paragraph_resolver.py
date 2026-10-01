from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from app.text_processing.classifiers.base import BlockRelationshipClassifier
from app.text_processing.content_graph import GraphResolution, resolve_components
from app.text_processing.feature_extractor import extract_features
from app.text_processing.feature_extractor import cosine_similarity
from app.text_processing.embeddings.base import SemanticEmbeddingUnavailable, TextEmbedder
from app.text_processing.group_resolver import resolve_groups
from app.text_processing.types import ContentGroup, RelationshipPrediction, TextBlock


@dataclass(frozen=True)
class ParagraphResolution:
    groups: list[ContentGroup]
    graph: GraphResolution
    predictions: tuple[RelationshipPrediction, ...]
    semantic_mode: str = "disabled"
    metrics: dict[str, float] | None = None


def resolve_paragraphs(
    blocks: list[TextBlock],
    classifier: BlockRelationshipClassifier,
    *,
    candidate_config: CandidateConfig | None = None,
    embedder: TextEmbedder | None = None,
    semantic_failure_policy: str = "fallback",
) -> ParagraphResolution:
    started = perf_counter()
    candidates = generate_candidates(blocks, candidate_config)
    by_id = {block.block_id: block for block in blocks}
    semantic_mode = "disabled"
    vectors: dict[str, tuple[float, ...]] = {}
    embedding_ms = 0.0
    if embedder is not None:
        embedding_started = perf_counter()
        try:
            batch = embedder.embed_many([block.normalized_text for block in blocks])
            vectors = {block.block_id: vector for block, vector in zip(blocks, batch.vectors, strict=True)}
            semantic_mode = "available"
        except (SemanticEmbeddingUnavailable, RuntimeError, ValueError):
            if semantic_failure_policy == "fail_fast":
                raise
            semantic_mode = "unavailable"
            vectors = {}
        embedding_ms = (perf_counter() - embedding_started) * 1000
    predictions = tuple(
        classifier.predict(
            by_id[candidate.block_a_id], by_id[candidate.block_b_id], candidate,
            extract_features(
                by_id[candidate.block_a_id], by_id[candidate.block_b_id], candidate,
                semantic_similarity=(cosine_similarity(vectors[candidate.block_a_id], vectors[candidate.block_b_id]) if vectors else None),
            ),
            semantic_available=bool(vectors),
        )
        for candidate in candidates
    )
    graph = resolve_components(blocks, list(predictions))
    candidate_by_id = {candidate.pair_id: candidate for candidate in candidates}
    confidence_by_edge: dict[tuple[str, str], float] = {}
    for prediction in graph.accepted:
        candidate = candidate_by_id[prediction.pair_id]
        confidence_by_edge[(candidate.block_a_id, candidate.block_b_id)] = prediction.probabilities[next(iter(prediction.probabilities))]
    groups = resolve_groups(blocks, graph.components, edge_confidences=confidence_by_edge)
    return ParagraphResolution(
        groups, graph, predictions, semantic_mode,
        {"embedding_time": embedding_ms, "pair_count": float(len(candidates)), "total_time": (perf_counter() - started) * 1000},
    )
