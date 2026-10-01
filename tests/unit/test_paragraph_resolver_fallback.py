import pytest

from app.core.config import Settings
from app.orchestration.extractor import ExtractionOrchestrator
from app.text_processing.classifiers.base import GroupingModelUnavailable
from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.text_processing.embeddings.base import SemanticEmbeddingUnavailable, SemanticIdentity, TextEmbedder
from tests.fixtures.grouping.factories import FakeClassifier, make_block


class FailingEmbedder(TextEmbedder):
    @property
    def identity(self):
        return SemanticIdentity("test", "missing", 2, "p", "n")

    def embed_many(self, texts):
        raise SemanticEmbeddingUnavailable("network", provider="test", model="missing", retryable=True)


def test_missing_model_is_typed():
    from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
    from app.text_processing.feature_extractor import feature_schema
    with pytest.raises(GroupingModelUnavailable):
        LightGBMRelationshipClassifier("missing-package", feature_schema())


def test_disabled_grouping_needs_no_optional_model():
    orchestrator = ExtractionOrchestrator(Settings(grouping_enabled=False))
    assert orchestrator.grouping_classifier is None


def test_semantic_failure_is_whole_document_fallback_or_typed_fail_fast():
    blocks = [make_block("a", "a"), make_block("b", "b", ordinal=1)]
    result = resolve_paragraphs(blocks, FakeClassifier(default=.9), embedder=FailingEmbedder(), semantic_failure_policy="fallback")
    assert result.semantic_mode == "unavailable"
    assert all(not prediction.semantic_available for prediction in result.predictions)
    with pytest.raises(SemanticEmbeddingUnavailable):
        resolve_paragraphs(blocks, FakeClassifier(default=.9), embedder=FailingEmbedder(), semantic_failure_policy="fail_fast")
