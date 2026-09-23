from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from redis import Redis

from app.core.config import Settings


_MEMORY_JOBS: dict[str, dict[str, Any]] = {}
_MEMORY_LOCK = RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Small durable state store used by API, worker and status polling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.job_store_backend.strip().lower()
        if self.backend not in {"redis", "memory"}:
            raise ValueError("job_store_backend must be 'redis' or 'memory'")
        self.redis = (
            Redis.from_url(settings.job_store_redis_url, decode_responses=True)
            if self.backend == "redis"
            else None
        )

    def _key(self, job_id: str) -> str:
        return f"wiki-hami:job:{job_id}"

    def create(self, job_id: str, document_id: str) -> dict[str, Any]:
        now = _now()
        record = {
            "job_id": job_id,
            "document_id": document_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "outputs": None,
            "error": None,
            "callback_delivered": None,
            "callback_error": None,
        }
        self._write(record)
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        if self.backend == "memory":
            with _MEMORY_LOCK:
                value = _MEMORY_JOBS.get(job_id)
                return dict(value) if value is not None else None
        assert self.redis is not None
        raw = self.redis.get(self._key(job_id))
        return json.loads(raw) if raw else None

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        record = self.get(job_id)
        if record is None:
            raise KeyError(f"unknown job_id: {job_id}")
        record.update(changes)
        record["updated_at"] = _now()
        self._write(record)
        return record

    def _write(self, record: dict[str, Any]) -> None:
        if self.backend == "memory":
            with _MEMORY_LOCK:
                _MEMORY_JOBS[record["job_id"]] = dict(record)
            return
        assert self.redis is not None
        self.redis.setex(
            self._key(record["job_id"]),
            self.settings.job_store_ttl_seconds,
            json.dumps(record, ensure_ascii=False),
        )
