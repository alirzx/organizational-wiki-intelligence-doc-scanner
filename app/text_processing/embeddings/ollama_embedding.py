from __future__ import annotations

import math
from time import perf_counter, sleep
from threading import RLock, Lock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Sequence

import httpx

from app.text_processing.embeddings.base import EmbeddingBatch, EmbeddingCache, EmbeddingProtocolError, SemanticEmbeddingUnavailable, SemanticIdentity, TextEmbedder
from app.text_processing.embeddings.cache import semantic_cache_key

_provider_lock = Lock()
_provider_executor = None
_provider_slots = BoundedSemaphore(1)


def close_provider_executor():
    global _provider_executor
    with _provider_lock:
        executor,_provider_executor = _provider_executor,None
    if executor is not None:
        executor.shutdown(wait=True,cancel_futures=True)


class OllamaEmbedder(TextEmbedder):
    def __init__(
        self, *, base_url: str, model: str = "embeddinggemma", dimensions: int = 768,
        timeout: float = 30, batch_size: int = 32, truncate: bool = False,
        keep_alive: str = "5m", max_attempts: int = 2, cache: EmbeddingCache | None = None,
        transport: httpx.BaseTransport | None = None,
        retry_backoff: float = .25, document_deadline: float = 60.,
    ):
        self._identity = SemanticIdentity("ollama", model, dimensions, "sentence_similarity_v1", "normalizer.v1")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.truncate = truncate
        self.keep_alive = keep_alive
        self.max_attempts = max_attempts
        self.cache = cache
        self.transport = transport
        self.retry_backoff = retry_backoff
        self.document_deadline = document_deadline
        self._client = None
        self._lock = RLock()
        if batch_size < 1 or max_attempts < 1 or timeout <= 0 or document_deadline <= 0 or retry_backoff < 0:
            raise ValueError('invalid embedding limits')

    def close(self):
        close_provider_executor()
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    @property
    def identity(self) -> SemanticIdentity:
        return self._identity

    @staticmethod
    def _prompt(text: str) -> str:
        return f"task: sentence similarity | query: {text}"

    def _post(self, payload, deadline):
        global _provider_executor
        remaining = deadline-perf_counter()
        if remaining <= 0 or not _provider_slots.acquire(timeout=max(0.,remaining)):
            raise SemanticEmbeddingUnavailable('deadline',provider='ollama',model=self.identity.model)
        try:
            with _provider_lock:
                if _provider_executor is None:
                    _provider_executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='ocr-embedding')
                if self._client is None:
                    self._client = httpx.Client(timeout=self.timeout,transport=self.transport)
                future = _provider_executor.submit(self._client.post,f'{self.base_url}/api/embed',json=payload,
                                                    timeout=min(self.timeout,max(.001,deadline-perf_counter())))
        except BaseException:
            _provider_slots.release()
            raise
        future.add_done_callback(lambda completed:_provider_slots.release())
        try:
            return future.result(timeout=max(0.,deadline-perf_counter()))
        except FutureTimeout as exc:
            future.cancel()
            raise SemanticEmbeddingUnavailable('deadline',provider='ollama',model=self.identity.model) from exc

    def _request(self, texts: list[str], deadline: float, diagnostics: dict) -> tuple[tuple[float, ...], ...]:
        payload = {
            "model": self.identity.model, "input": [self._prompt(text) for text in texts],
            "truncate": self.truncate, "dimensions": self.identity.dimensions, "keep_alive": self.keep_alive,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            if deadline <= perf_counter():
                raise SemanticEmbeddingUnavailable('deadline',provider='ollama',model=self.identity.model)
            if attempt:
                backoff = self.retry_backoff*2**(attempt-1)
                if perf_counter()+backoff >= deadline:
                    raise SemanticEmbeddingUnavailable('deadline',provider='ollama',model=self.identity.model)
                sleep(backoff)
                diagnostics['retry_count'] += 1
            try:
                response = self._post(payload,deadline)
                if response.status_code >= 400:
                    retryable = response.status_code >= 500 or response.status_code == 429
                    error = SemanticEmbeddingUnavailable("http_status", provider="ollama", model=self.identity.model, status=response.status_code, retryable=retryable)
                    if not retryable or attempt + 1 >= self.max_attempts:
                        raise error
                    last_error = error
                    continue
                data = response.json()
                raw = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(raw, list) or len(raw) != len(texts):
                    raise EmbeddingProtocolError("embedding response count mismatch")
                if any(not isinstance(vector,list) for vector in raw):
                    raise EmbeddingProtocolError('embedding vectors must be arrays')
                vectors = tuple(tuple(float(value) for value in vector) for vector in raw)
                if any(len(vector) != self.identity.dimensions or any(not math.isfinite(value) for value in vector) for vector in vectors):
                    raise EmbeddingProtocolError("embedding response dimension or value invalid")
                return vectors
            except httpx.RequestError as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise SemanticEmbeddingUnavailable("network", provider="ollama", model=self.identity.model, retryable=True) from exc
            except (ValueError,TypeError) as exc:
                raise EmbeddingProtocolError("embedding response is not valid JSON") from exc
        raise SemanticEmbeddingUnavailable("network", provider="ollama", model=self.identity.model, retryable=True) from last_error

    def embed_many(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts or any(not text for text in texts):
            raise ValueError("embedding inputs must be non-empty")
        started = perf_counter()
        deadline = started+self.document_deadline
        if not self._lock.acquire(timeout=self.document_deadline):
            raise SemanticEmbeddingUnavailable('deadline',provider='ollama',model=self.identity.model)
        try:
            return self._embed_many(texts,started,deadline)
        finally:
            self._lock.release()

    def _embed_many(self,texts,started,deadline):
        diagnostics = {'batch_count':0,'retry_count':0}
        keys = [semantic_cache_key(text, self.identity) for text in texts]
        cached = self.cache.get_many(keys) if self.cache else {}
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, value in zip(keys, texts, strict=True):
            if key not in cached and key not in seen:
                seen.add(key); missing_keys.append(key); missing_texts.append(value)
        created: dict[str, tuple[float, ...]] = {}
        try:
            for offset in range(0, len(missing_texts), self.batch_size):
                batch_texts = missing_texts[offset:offset + self.batch_size]
                batch_keys = missing_keys[offset:offset + self.batch_size]
                diagnostics['batch_count'] += 1
                validated = dict(zip(batch_keys,self._request(batch_texts,deadline,diagnostics),strict=True))
                created.update(validated)
                if self.cache:
                    self.cache.set_many(validated)
        except (SemanticEmbeddingUnavailable,EmbeddingProtocolError) as exc:
            exc.diagnostics = {**diagnostics,'unique_texts':len(set(keys)),
                               'cache_hits':sum(key in cached for key in keys),'cache_misses':len(missing_keys),
                               'provider_latency_ms':(perf_counter()-started)*1000}
            raise
        all_values = cached | created
        return EmbeddingBatch(
            identity=self.identity, vectors=tuple(all_values[key] for key in keys), requested_count=len(texts),
            unique_count=len(set(keys)), cache_hit_count=sum(key in cached for key in keys),
            provider_timings={"embedding_time": (perf_counter() - started) * 1000, **diagnostics,
                             'provider_latency_ms':(perf_counter()-started)*1000,
                             'unique_cache_misses':len(missing_keys)},
        )
