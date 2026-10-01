from app.text_processing.embeddings.base import SemanticIdentity
from app.text_processing.embeddings.cache import RunEmbeddingCache, semantic_cache_key


def test_cache_keys_hide_text_include_identity_and_evict_lru():
    identity = SemanticIdentity("ollama", "m", 2, "p", "n")
    key = semantic_cache_key("secret text", identity)
    assert "secret" not in key and len(key) == 64
    assert key != semantic_cache_key("secret text", SemanticIdentity("ollama", "m2", 2, "p", "n"))
    cache = RunEmbeddingCache(1)
    cache.set_many({key: (1., 2.)})
    second = semantic_cache_key("other", identity)
    cache.set_many({second: (2., 3.)})
    assert cache.get_many([key, second]) == {second: (2., 3.)}
