import time
import httpx
import pytest

from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.text_processing.embeddings.ollama_embedding import OllamaEmbedder
from app.text_processing.embeddings.cache import RunEmbeddingCache
from app.text_processing.embeddings.base import SemanticEmbeddingUnavailable
from tests.fixtures.grouping.factories import FakeClassifier, make_block


def test_successful_batches_survive_later_failure_and_client_is_reused():
    calls = []
    fail = True
    def handler(request):
        nonlocal fail
        body = __import__('json').loads(request.content)
        text = body['input'][0]
        calls.append(text)
        if text.endswith('b') and fail:
            return httpx.Response(503)
        return httpx.Response(200,json={'embeddings':[[1.,0.]]})
    embedder = OllamaEmbedder(base_url='http://test',dimensions=2,batch_size=1,max_attempts=1,
                             cache=RunEmbeddingCache(10),transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(SemanticEmbeddingUnavailable):
            embedder.embed_many(['a','b'])
        client = embedder._client
        fail = False
        result = embedder.embed_many(['a','b'])
        assert result.cache_hit_count == 1 and result.provider_timings['batch_count'] == 1
        assert len(calls) == 3 and embedder._client is client
    finally:
        embedder.close()


def test_deadline_bounds_slow_provider_without_creating_extra_requests():
    calls = []
    def handler(request):
        calls.append(1)
        time.sleep(.15)
        return httpx.Response(200,json={'embeddings':[[1.,0.]]})
    embedder = OllamaEmbedder(base_url='http://test',dimensions=2,document_deadline=.03,
                             transport=httpx.MockTransport(handler))
    try:
        started = time.perf_counter()
        with pytest.raises(SemanticEmbeddingUnavailable,match='deadline'):
            embedder.embed_many(['a'])
        assert time.perf_counter()-started < .12
        assert len(calls) == 1
    finally:
        embedder.close()  # Wait for bounded in-flight work before closing its client.


def test_retry_count_and_isolated_blocks_do_not_require_embeddings():
    calls = []
    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200,json={'embeddings':[[1.,0.],[1.,0.]]})
    embedder = OllamaEmbedder(base_url='http://test',dimensions=2,retry_backoff=0,transport=httpx.MockTransport(handler))
    try:
        isolated = resolve_paragraphs([make_block('only','text')],FakeClassifier(),embedder=embedder)
        assert not calls and isolated.semantic_mode == 'disabled'
        blocks = [make_block('a','unfinished'),make_block('b','continued',ordinal=1)]
        result = resolve_paragraphs(blocks,FakeClassifier(default=.9),embedder=embedder)
        assert result.embedding_diagnostics['retry_count'] == 1
        assert result.semantic_mode == 'available' and len(result.groups) == 1
    finally:
        embedder.close()
