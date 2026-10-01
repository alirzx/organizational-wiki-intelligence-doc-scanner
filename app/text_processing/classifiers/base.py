from __future__ import annotations

from abc import ABC, abstractmethod

from app.text_processing.types import CandidatePair, FeatureVector, RelationshipPrediction, TextBlock


class GroupingModelError(RuntimeError):
    """Base error for model discovery, validation, load, or prediction failures."""


class GroupingModelUnavailable(GroupingModelError):
    pass


class GroupingModelIncompatible(GroupingModelError):
    pass


class BlockRelationshipClassifier(ABC):
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

