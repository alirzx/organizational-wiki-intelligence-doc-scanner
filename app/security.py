from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException

from app.core.config import get_settings


def verify_backend_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Authenticate Backend -> AI production calls when a key is configured."""
    expected = get_settings().backend_api_key
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
