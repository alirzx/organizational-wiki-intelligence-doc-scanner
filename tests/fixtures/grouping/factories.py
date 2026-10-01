from __future__ import annotations

from collections.abc import Sequence

from app.schemas.common import BBox
from app.text_processing.classifiers.base import BlockRelationshipClassifier
from app.text_processing.embeddings.base import EmbeddingBatch, SemanticIdentity, TextEmbedder
from app.text_processing.ids import stable_pair_id
from app.text_processing.types import (
    CandidatePair,
    CandidateReason,
    RelationshipDecision,
    RelationshipLabel,
    RelationshipPrediction,
    FeatureVector,
    PageSpan,
    TextBlock,
)


def make_block(
    block_id: str,
    text: str,
    *,
    page_number: int = 1,
    ordinal: int = 0,
    x1: float = 100,
    y1: float = 100,
    x2: float = 500,
    y2: float = 130,
    column_id: str | None = "main",
) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        document_id="doc-test",
        page_id=f"doc-test:p{page_number}",
        page_number=page_number,
        content_ordinal=ordinal,
        original_text=text,
        normalized_text=text,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        page_width=1000,
        page_height=1400,
        ocr_confidence=0.95,
        column_id=column_id,
    )


def make_candidate(
    block_a: TextBlock,
    block_b: TextBlock,
    *reasons: CandidateReason,
) -> CandidatePair:
    selected = reasons or (CandidateReason.READING_WINDOW,)
    return CandidatePair(
        pair_id=stable_pair_id(block_a.block_id, block_b.block_id, "candidate.v1"),
        block_a_id=block_a.block_id,
        block_b_id=block_b.block_id,
        cross_page=block_a.page_number != block_b.page_number,
        page_distance=block_b.page_number - block_a.page_number,
        reasons=tuple(selected),
    )


def make_prediction(
    candidate: CandidatePair,
    probability_same: float,
    *,
    decision: RelationshipDecision | None = None,
    semantic_available: bool = False,
) -> RelationshipPrediction:
    selected_decision = decision or (
        RelationshipDecision.MERGE
        if probability_same >= 0.75
        else RelationshipDecision.SEPARATE
    )
    label = (
        RelationshipLabel.SAME_GROUP
        if probability_same >= 0.5
        else RelationshipLabel.NEW_GROUP
    )
    probabilities = {
        RelationshipLabel.SAME_GROUP: probability_same,
        RelationshipLabel.NEW_GROUP: 1 - probability_same,
    }
    return RelationshipPrediction(
        pair_id=candidate.pair_id,
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
        decision=selected_decision,
        semantic_available=semantic_available,
        model_package_id="fixture-model",
    )


def make_page_span(*blocks: TextBlock) -> PageSpan:
    if not blocks:
        raise ValueError("at least one block is required")
    return PageSpan(
        page_id=blocks[0].page_id,
        page_number=blocks[0].page_number,
        bbox=BBox(
            x1=min(block.bbox.x1 for block in blocks),
            y1=min(block.bbox.y1 for block in blocks),
            x2=max(block.bbox.x2 for block in blocks),
            y2=max(block.bbox.y2 for block in blocks),
        ),
        member_ids=tuple(block.block_id for block in blocks),
        text="\n".join(block.normalized_text for block in blocks),
        raw_text="\n".join(block.original_text for block in blocks),
    )


class FakeClassifier(BlockRelationshipClassifier):
    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.1):
        self.scores = scores or {}
        self.default = default

    @property
    def package_id(self) -> str:
        return "fixture-model"

    def predict(
        self,
        block_a: TextBlock,
        block_b: TextBlock,
        candidate: CandidatePair,
        features: FeatureVector,
        *,
        semantic_available: bool,
    ) -> RelationshipPrediction:
        return make_prediction(
            candidate,
            self.scores.get(candidate.pair_id, self.default),
            semantic_available=semantic_available,
        )


class FakeEmbedder(TextEmbedder):
    def __init__(self, dimensions: int = 4):
        self._identity = SemanticIdentity(
            provider="fixture",
            model="deterministic",
            dimensions=dimensions,
            prompt_profile="none",
            normalizer_version="normalizer.v1",
        )

    @property
    def identity(self) -> SemanticIdentity:
        return self._identity

    def embed_many(self, texts: Sequence[str]) -> EmbeddingBatch:
        vectors = tuple(
            tuple(float((sum(map(ord, text)) + offset) % 17) / 17 for offset in range(self.identity.dimensions))
            for text in texts
        )
        return EmbeddingBatch(
            identity=self.identity,
            vectors=vectors,
            requested_count=len(texts),
            unique_count=len(set(texts)),
        )
