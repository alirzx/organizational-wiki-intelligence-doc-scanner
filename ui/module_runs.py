"""UI-only helpers for displaying independent module inspection runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


ALL_MODULES = ("ocr", "figure_table", "stamp_signature")
OBJECT_TYPES = ("paragraph", "table", "figure", "stamp", "signature")
MODULE_ENDPOINTS = {
    "ocr": "/ocr",
    "figure_table": "/figure-table",
    "stamp_signature": "/stamp-signature",
}
MODULE_LABELS = {
    "ocr": "OCR",
    "figure_table": "Figure/Table",
    "stamp_signature": "Stamp/Signature",
}


def build_module_request(document_id: str, page: Mapping[str, Any]) -> dict[str, Any]:
    """Map one existing MinIO page descriptor to ModuleImageRequest JSON."""
    return {
        "document_id": document_id,
        "image_url": page["image_url"],
        "page_number": page["page_number"],
        "page_id": page["page_id"],
        "page_metadata": page.get("metadata") or {},
    }


def build_module_document_result(
    module_pages: Sequence[Mapping[str, Any]],
    *,
    document_metadata: Mapping[str, Any],
    page_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a document-shaped UI result from canonical module responses."""
    if not module_pages:
        raise ValueError("At least one module page response is required")

    modules = {str(page["module"]) for page in module_pages}
    document_ids = {str(page["document_id"]) for page in module_pages}
    if len(modules) != 1:
        raise ValueError("Module page responses must all belong to one module")
    if len(document_ids) != 1:
        raise ValueError("Module page responses must all belong to one document")

    module = next(iter(modules))
    metadata_by_page = page_metadata or {}
    pages: list[dict[str, Any]] = []
    all_objects: list[dict[str, Any]] = []
    states: list[str] = []
    warnings: list[str] = []
    duration_ms = 0.0

    for response in sorted(module_pages, key=lambda item: int(item["page_number"])):
        status = dict(response["status"])
        state = str(status.get("state", "failed"))
        page_warnings = list(status.get("warnings") or [])
        page_objects = [dict(item) for item in response.get("objects") or []]
        page_objects.sort(
            key=lambda item: (
                float((item.get("bbox") or {}).get("y1", 0)),
                float((item.get("bbox") or {}).get("x1", 0)),
                str(item.get("type", "")),
            )
        )
        page_id = str(response["page_id"])
        page_duration = float(status.get("duration_ms") or 0)
        pages.append(
            {
                "schema_version": response["schema_version"],
                "request_id": response["request_id"],
                "document_id": response["document_id"],
                "page_id": page_id,
                "page_number": response["page_number"],
                "page_metadata": dict(metadata_by_page.get(page_id) or {}),
                "image": response["image"],
                "transform": response["transform"],
                "objects": page_objects,
                "modules": {module: status},
                "processing": {
                    "state": state,
                    "duration_ms": page_duration,
                    "warnings": page_warnings,
                },
            }
        )
        all_objects.extend(page_objects)
        states.append(state)
        warnings.extend(page_warnings)
        duration_ms += page_duration

    all_objects.sort(
        key=lambda item: (
            int(item.get("page_number", 0)),
            float((item.get("bbox") or {}).get("y1", 0)),
            float((item.get("bbox") or {}).get("x1", 0)),
            str(item.get("type", "")),
        )
    )
    if all(state == "success" for state in states):
        document_state = "success"
    elif all(state == "failed" for state in states):
        document_state = "failed"
    else:
        document_state = "partial_success"

    counts = Counter(str(item.get("type", "")) for item in all_objects)
    first = module_pages[0]
    return {
        "schema_version": first["schema_version"],
        "request_id": first["request_id"],
        "document_id": first["document_id"],
        "document_metadata": dict(document_metadata),
        "page_count": len(pages),
        "pages": pages,
        "objects": all_objects,
        "object_counts": {object_type: counts[object_type] for object_type in OBJECT_TYPES},
        "processing": {
            "state": document_state,
            "duration_ms": duration_ms,
            "warnings": warnings,
        },
    }


def success_message(executed_modules: Sequence[str], *, persisted: bool) -> str:
    executed = set(executed_modules)
    if executed == set(ALL_MODULES):
        message = "All modules succeeded."
    elif len(executed) == 1:
        message = f"{MODULE_LABELS[next(iter(executed))]} succeeded on all pages."
    else:
        message = "Selected modules succeeded."
    if persisted:
        message += " Product artifacts were persisted to MinIO."
    return message
