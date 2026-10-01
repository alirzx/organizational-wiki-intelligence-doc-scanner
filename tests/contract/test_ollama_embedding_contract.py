import httpx
import pytest

from app.text_processing.embeddings.base import EmbeddingProtocolError
from app.text_processing.embeddings.cache import RunEmbeddingCache
from app.text_processing.embeddings.ollama_embedding import OllamaEmbedder


def test_ollama_batch_payload_order_and_cache_reconstruction():
    payloads = []
    def handler(request):
        payload = __import__("json").loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, json={"embeddings": [[float(i), 0.] for i, _ in enumerate(payload["input"], 1)]})
    embedder = OllamaEmbedder(base_url="http://test", dimensions=2, batch_size=2, cache=RunEmbeddingCache(8), transport=httpx.MockTransport(handler))
    first = embedder.embed_many(["a", "b", "a"])
    second = embedder.embed_many(["a", "b", "a"])
    assert first.vectors == ((1., 0.), (2., 0.), (1., 0.))
    assert second.cache_hit_count == 3 and len(payloads) == 1
    assert payloads[0]["truncate"] is False and payloads[0]["dimensions"] == 2
    assert payloads[0]["input"][0].startswith("task: sentence similarity | query:")


@pytest.mark.parametrize("body", [{}, {"embeddings": [[1.]]}, {"embeddings": [[float("nan"), 0.]]}])
def test_malformed_responses_are_rejected(body):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    with pytest.raises(EmbeddingProtocolError):
        OllamaEmbedder(base_url="http://test", dimensions=2, transport=transport).embed_many(["a"])
