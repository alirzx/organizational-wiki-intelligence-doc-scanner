from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from app.text_processing.candidate_generator import CandidateConfig, generate_candidate_batch
from app.text_processing.classifiers.base import BlockRelationshipClassifier
from app.text_processing.content_graph import GraphResolution, resolve_components
from app.text_processing.feature_extractor import extract_features, precompute_block_features
from app.text_processing.embeddings.base import SemanticEmbeddingUnavailable, TextEmbedder
from app.text_processing.group_resolver import resolve_groups
from app.text_processing.types import ContentGroup, GroupingMode, RelationshipPrediction, RelationshipLabel, TextBlock
import math


@dataclass(frozen=True)
class ParagraphResolution:
    groups: list[ContentGroup]
    graph: GraphResolution
    predictions: tuple[RelationshipPrediction, ...]
    semantic_mode: str = "disabled"
    metrics: dict[str, float] | None = None
    candidate_diagnostics: dict | None = None
    embedding_diagnostics: dict | None = None


def resolve_paragraphs(
    blocks: list[TextBlock],
    classifier: BlockRelationshipClassifier,
    *,
    candidate_config: CandidateConfig | None = None,
    embedder: TextEmbedder | None = None,
    semantic_failure_policy: str = "fallback",
) -> ParagraphResolution:
    started = perf_counter()
    candidate_config = candidate_config or CandidateConfig()
    if classifier.grouping_mode == GroupingMode.CLUSTERED:
        candidate_config = replace(candidate_config, adjacent_only=True)
    generation = generate_candidate_batch(blocks, candidate_config)
    candidates = generation.pairs
    candidate_ms = (perf_counter()-started)*1000
    by_id = {block.block_id: block for block in blocks}
    semantic_mode = "disabled"
    vectors: dict[str, tuple[float, ...]] = {}
    embedding_ms = 0.0
    embedding_diagnostics = {}
    needed_ids = {item for c in candidates for item in (c.block_a_id,c.block_b_id)}
    needed = [b for b in blocks if b.block_id in needed_ids and b.normalized_text]
    if embedder is not None and needed:
        embedding_started = perf_counter()
        try:
            batch = embedder.embed_many([block.normalized_text for block in needed])
            vectors = {block.block_id: vector for block, vector in zip(needed, batch.vectors, strict=True)}
            embedding_diagnostics = {**batch.identity.__dict__, 'unique_texts':batch.unique_count,
                                     'cache_hits':batch.cache_hit_count,
                                     'cache_misses':batch.requested_count-batch.cache_hit_count,
                                     **batch.provider_timings}
            semantic_mode = "available"
        except (SemanticEmbeddingUnavailable, RuntimeError, ValueError) as exc:
            if semantic_failure_policy == "fail_fast":
                raise
            semantic_mode = "unavailable"
            vectors = {}
            embedding_diagnostics = {**embedder.identity.__dict__, 'failure_reason':getattr(exc,'kind',type(exc).__name__),
                                     **getattr(exc,'diagnostics',{})}
        embedding_ms = (perf_counter() - embedding_started) * 1000
    feature_started = perf_counter()
    block_features = precompute_block_features(blocks)
    norms = {key:math.sqrt(sum(x*x for x in vector)) or 1. for key,vector in vectors.items()}
    unit_vectors = {key:tuple(value/norms[key] for value in vector) for key,vector in vectors.items()}
    def similarity(candidate):
        if candidate.block_a_id not in unit_vectors or candidate.block_b_id not in unit_vectors:
            return None
        return max(-1.,min(1.,sum(x*y for x,y in zip(unit_vectors[candidate.block_a_id],unit_vectors[candidate.block_b_id],strict=True))))
    features = {
        candidate.pair_id: extract_features(
            by_id[candidate.block_a_id], by_id[candidate.block_b_id], candidate,
            semantic_similarity=similarity(candidate), block_features=block_features,
        ) for candidate in candidates
    }
    feature_ms = (perf_counter()-feature_started)*1000
    classify_started = perf_counter()
    classifier.prepare(blocks, candidates, features)
    predictions = tuple(classifier.predict_many(by_id,candidates,features,semantic_available=bool(vectors)))
    classifier_ms = (perf_counter()-classify_started)*1000
    graph_started = perf_counter()
    graph = resolve_components(blocks, list(predictions), candidates=candidates)
    graph_ms = (perf_counter()-graph_started)*1000
    candidate_by_id = {candidate.pair_id: candidate for candidate in candidates}
    confidence_by_edge: dict[tuple[str, str], float] = {}
    for prediction in graph.accepted:
        candidate = candidate_by_id[prediction.pair_id]
        confidence_by_edge[(candidate.block_a_id, candidate.block_b_id)] = prediction.probabilities[RelationshipLabel.SAME_GROUP]
    groups_started = perf_counter()
    groups = resolve_groups(blocks, graph.components, edge_confidences=confidence_by_edge)
    if classifier.grouping_mode == GroupingMode.CLUSTERED:
        groups = [replace(group, confidence_method="minimum_clustering_affinity",
                          metadata={**group.metadata, "algorithm": "dbscan", "confidence_kind": "affinity"})
                  for group in groups]
    return ParagraphResolution(
        groups, graph, predictions, semantic_mode,
        {"candidate_time":candidate_ms, "embedding_time": embedding_ms, 'feature_time':feature_ms,
         'classifier_time':classifier_ms, 'graph_time':graph_ms, 'group_time':(perf_counter()-groups_started)*1000,
         "pair_count": float(len(candidates)), "total_time": (perf_counter() - started) * 1000},
        generation.diagnostics, embedding_diagnostics,
    )
