from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable

from app.schemas.common import BBox


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{sha256(encoded).hexdigest()[:24]}"


def stable_block_id(
    *,
    document_id: str,
    page_id: str,
    bbox: BBox,
    content_ordinal: int,
    text: str,
    collision: int = 0,
) -> str:
    return _stable_id(
        "blk",
        {
            "document_id": document_id,
            "page_id": page_id,
            "bbox": bbox.model_dump(mode="json"),
            "content_ordinal": content_ordinal,
            "text": text,
            "collision": collision,
        },
    )


def stable_pair_id(block_a_id: str, block_b_id: str, schema_version: str) -> str:
    if block_a_id == block_b_id:
        raise ValueError("pair blocks must be distinct")
    return _stable_id(
        "pair",
        {"a": block_a_id, "b": block_b_id, "schema_version": schema_version},
    )


def stable_group_id(member_ids: Iterable[str], schema_version: str) -> str:
    ordered = list(member_ids)
    if not ordered:
        raise ValueError("group members must not be empty")
    return _stable_id("grp", {"members": ordered, "schema_version": schema_version})

