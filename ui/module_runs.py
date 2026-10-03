"""UI-only helpers for displaying independent module inspection runs."""

from __future__ import annotations

from collections.abc import Sequence


ALL_MODULES = ("ocr", "figure_table", "stamp_signature")
MODULE_LABELS = {
    "ocr": "OCR",
    "figure_table": "Figure/Table",
    "stamp_signature": "Stamp/Signature",
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
