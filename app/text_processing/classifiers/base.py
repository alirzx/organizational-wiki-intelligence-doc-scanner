from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from app.text_processing.types import CandidatePair, FeatureVector, GroupingMode, RelationshipPrediction, TextBlock


class GroupingModelError(RuntimeError):
    """Base error for model discovery, validation, load, or prediction failures."""


class GroupingModelUnavailable(GroupingModelError):
    pass


class GroupingModelIncompatible(GroupingModelError):
    pass


class BlockRelationshipClassifier(ABC):
    grouping_mode = GroupingMode.LEARNED

    def predict_many(self, blocks: dict[str, TextBlock], candidates: list[CandidatePair],
                     features: dict[str, FeatureVector], *, semantic_available: bool) -> list[RelationshipPrediction]:
        from app.text_processing.feature_extractor import feature_schema
        index = feature_schema().semantic_indices[1]
        return [replace(self.predict(blocks[c.block_a_id], blocks[c.block_b_id], c, features[c.pair_id],
                                     semantic_available=bool(features[c.pair_id].values[index])), block_a_id=c.block_a_id,
                        block_b_id=c.block_b_id) for c in candidates]

    def prepare(
        self, blocks: list[TextBlock], candidates: list[CandidatePair],
        features: dict[str, FeatureVector],
    ) -> None:
        """Prepare document-scoped state; trained pair classifiers need none."""

    @property
    @abstractmethod
    def package_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        block_a: TextBlock,
        block_b: TextBlock,
        candidate: CandidatePair,
        features: FeatureVector,
        *,
        semantic_available: bool,
    ) -> RelationshipPrediction:
        raise NotImplementedError

