import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_grouping_and_semantic_switches_are_independent():
    assert not Settings(_env_file=None, grouping_enabled=True).semantic_features_enabled
    assert not Settings(_env_file=None, semantic_features_enabled=True).grouping_enabled


@pytest.mark.parametrize("values", [
    {"grouping_uncertain_lower": .8, "grouping_merge_threshold": .7},
    {"ollama_embedding_timeout_seconds": 0}, {"ollama_embedding_batch_size": 0},
    {"embedding_cache_max_entries": 0}, {"ollama_embedding_dimensions": 42},
    {"grouping_backend": "unknown"}, {"grouping_cluster_eps": 0},
    {"grouping_cluster_eps": 1.1}, {"grouping_cluster_min_samples": 1},
    {'grouping_workers':0}, {'grouping_prediction_batch_size':0}, {'grouping_inference_threads':0},
    {'embedding_document_deadline_seconds':0}, {'embedding_retry_backoff_seconds':-1},
])
def test_invalid_grouping_settings_are_rejected(values):
    with pytest.raises(ValidationError):
        Settings(**values)


def test_standard_ollama_aliases(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "model-x")
    settings = Settings()
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.ollama_embedding_model == "model-x"


def test_partial_lightgbm_threshold_override_defers_order_to_package():
    settings = Settings(_env_file=None,grouping_backend='lightgbm',grouping_merge_threshold=.5)
    assert settings.grouping_merge_threshold == .5
    assert 'grouping_uncertain_lower' not in settings.model_fields_set
    with pytest.raises(ValidationError):
        Settings(_env_file=None,grouping_backend='clustering',grouping_merge_threshold=.5)
