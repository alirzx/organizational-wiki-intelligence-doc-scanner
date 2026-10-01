from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
from typing import Sequence

from app.text_processing.embeddings.base import EmbeddingCache, SemanticIdentity


def semantic_cache_key(text: str, identity: SemanticIdentity) -> str:
    payload = json.dumps({"text": text, "identity": identity.__dict__}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


class RunEmbeddingCache(EmbeddingCache):
    def __init__(self, max_entries: int):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    def get_many(self, keys: Sequence[str]) -> dict[str, tuple[float, ...]]:
        found: dict[str, tuple[float, ...]] = {}
        for key in keys:
            if key in self._values:
                found[key] = self._values.pop(key)
                self._values[key] = found[key]
        return found

    def set_many(self, values: dict[str, tuple[float, ...]]) -> None:
        for key, vector in values.items():
            self._values.pop(key, None)
            self._values[key] = vector
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
