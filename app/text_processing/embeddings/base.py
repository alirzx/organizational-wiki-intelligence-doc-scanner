from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Sequence


@dataclass(frozen=True)
class SemanticIdentity:
    provider: str
    model: str
    dimensions: int
    prompt_profile: str
    normalizer_version: str
    adapter_version: str = "embed.v1"
    model_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.model or self.dimensions <= 0:
            raise ValueError("semantic identity requires provider, model, and positive dimensions")


@dataclass(frozen=True)
class EmbeddingBatch:
    identity: SemanticIdentity
    vectors: tuple[tuple[float, ...], ...]
    requested_count: int
    unique_count: int
    cache_hit_count: int = 0
    provider_timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.requested_count < 1 or self.unique_count < 1 or self.unique_count > self.requested_count:
            raise ValueError("embedding batches require non-empty inputs and valid unique count")
        if self.requested_count != len(self.vectors):
            raise ValueError("embedding response cardinality must match requested inputs")
        if not 0 <= self.cache_hit_count <= self.requested_count:
            raise ValueError("cache hit count is invalid")
        for vector in self.vectors:
            if len(vector) != self.identity.dimensions:
                raise ValueError("embedding dimension mismatch")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("embedding vectors must be finite")


class SemanticEmbeddingUnavailable(RuntimeError):
    def __init__(
        self,
        kind: str,
        *,
        provider: str,
        model: str,
        status: int | None = None,
        retryable: bool = False,
    ):
        self.kind = kind
        self.provider = provider
        self.model = model
        self.status = status
        self.retryable = retryable
        super().__init__(f"semantic embedding unavailable: {kind}")


class EmbeddingProtocolError(RuntimeError):
    pass


class TextEmbedder(ABC):
    @property
    @abstractmethod
    def identity(self) -> SemanticIdentity:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        return list(self.embed_many([text]).vectors[0])

    @abstractmethod
    def embed_many(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise NotImplementedError


class EmbeddingCache(ABC):
    @abstractmethod
    def get_many(self, keys: Sequence[str]) -> dict[str, tuple[float, ...]]:
        raise NotImplementedError

    @abstractmethod
    def set_many(self, values: dict[str, tuple[float, ...]]) -> None:
        raise NotImplementedError
