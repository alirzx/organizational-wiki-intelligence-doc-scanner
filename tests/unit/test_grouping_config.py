import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_grouping_and_semantic_switches_are_independent():
    assert not Settings(grouping_enabled=True).semantic_features_enabled
    assert not Settings(semantic_features_enabled=True).grouping_enabled


@pytest.mark.parametrize("values", [
    {"grouping_uncertain_lower": .8, "grouping_merge_threshold": .7},
    {"ollama_embedding_timeout_seconds": 0}, {"ollama_embedding_batch_size": 0},
    {"embedding_cache_max_entries": 0}, {"ollama_embedding_dimensions": 42},
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
