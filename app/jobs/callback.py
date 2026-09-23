from __future__ import annotations

import time
from typing import Any

import requests

from app.core.config import Settings


class CallbackDeliveryError(RuntimeError):
    pass


def deliver_callback(settings: Settings, payload: dict[str, Any]) -> None:
    """Deliver a terminal job callback with bounded retry/backoff.

    No callback is attempted when callback_url is empty, which keeps local
    engineering use convenient. Production should always configure it.
    """
    if not settings.callback_url:
        return

    headers = {"Content-Type": "application/json"}
    if settings.callback_token:
        headers["Authorization"] = f"Bearer {settings.callback_token}"

    last_error: Exception | None = None
    attempts = max(1, settings.callback_max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                settings.callback_url,
                json=payload,
                headers=headers,
                timeout=settings.callback_timeout_seconds,
            )
            response.raise_for_status()
            return
        except Exception as exc:  # requests exposes several transport/status failures
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise CallbackDeliveryError(str(last_error) if last_error else "callback delivery failed")
