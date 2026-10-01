from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

import httpx

from app.text_processing.embeddings.base import EmbeddingBatch, EmbeddingCache, EmbeddingProtocolError, SemanticEmbeddingUnavailable, SemanticIdentity, TextEmbedder
from app.text_processing.embeddings.cache import semantic_cache_key


class OllamaEmbedder(TextEmbedder):
    def __init__(
        self, *, base_url: str, model: str = "embeddinggemma", dimensions: int = 768,
        timeout: float = 30, batch_size: int = 32, truncate: bool = False,
        keep_alive: str = "5m", max_attempts: int = 2, cache: EmbeddingCache | None = None,
        transport: httpx.BaseTransport | None = None,
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

    @property
    def identity(self) -> SemanticIdentity:
        return self._identity

    @staticmethod
    def _prompt(text: str) -> str:
        return f"task: sentence similarity | query: {text}"

    def _request(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        payload = {
            "model": self.identity.model, "input": [self._prompt(text) for text in texts],
            "truncate": self.truncate, "dimensions": self.identity.dimensions, "keep_alive": self.keep_alive,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                    response = client.post(f"{self.base_url}/api/embed", json=payload)
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
                vectors = tuple(tuple(float(value) for value in vector) for vector in raw)
                if any(len(vector) != self.identity.dimensions or any(not math.isfinite(value) for value in vector) for vector in vectors):
                    raise EmbeddingProtocolError("embedding response dimension or value invalid")
                return vectors
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise SemanticEmbeddingUnavailable("network", provider="ollama", model=self.identity.model, retryable=True) from exc
            except ValueError as exc:
                raise EmbeddingProtocolError("embedding response is not valid JSON") from exc
        raise SemanticEmbeddingUnavailable("network", provider="ollama", model=self.identity.model, retryable=True) from last_error

    def embed_many(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts or any(not text for text in texts):
            raise ValueError("embedding inputs must be non-empty")
        started = perf_counter()
        keys = [semantic_cache_key(text, self.identity) for text in texts]
        cached = self.cache.get_many(keys) if self.cache else {}
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, value in zip(keys, texts, strict=True):
            if key not in cached and key not in seen:
                seen.add(key); missing_keys.append(key); missing_texts.append(value)
        created: dict[str, tuple[float, ...]] = {}
        for offset in range(0, len(missing_texts), self.batch_size):
            batch_texts = missing_texts[offset:offset + self.batch_size]
            batch_keys = missing_keys[offset:offset + self.batch_size]
            created.update(zip(batch_keys, self._request(batch_texts), strict=True))
        if self.cache and created:
            self.cache.set_many(created)
        all_values = cached | created
        return EmbeddingBatch(
            identity=self.identity, vectors=tuple(all_values[key] for key in keys), requested_count=len(texts),
            unique_count=len(set(keys)), cache_hit_count=sum(key in cached for key in keys),
            provider_timings={"embedding_time": (perf_counter() - started) * 1000},
        )
