from __future__ import annotations

import re

from app.text_processing.structural_features import is_list_start
from app.text_processing.types import TextBlock


_COMPLETE = re.compile(r"[.!?؟؛;:]\s*$")
_CONNECTORS = re.compile(r"^(?:and|or|but|because|therefore|however|و|یا|اما|که|زیرا|بنابراین)\b", re.I)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text, re.UNICODE)


def extract_linguistic_features(a: TextBlock, b: TextBlock) -> dict[str, float]:
    a_len, b_len = len(a.normalized_text), len(b.normalized_text)
    return {
        "a_sentence_complete": float(bool(_COMPLETE.search(a.normalized_text))),
        "b_sentence_complete": float(bool(_COMPLETE.search(b.normalized_text))),
        "b_continuation_start": float(bool(_CONNECTORS.search(b.normalized_text))),
        "a_list_start_linguistic": float(is_list_start(a.normalized_text)),
        "b_list_start_linguistic": float(is_list_start(b.normalized_text)),
        "a_token_count": float(len(_tokens(a.normalized_text))),
        "b_token_count": float(len(_tokens(b.normalized_text))),
        "length_ratio": max(a_len, b_len) / max(1, min(a_len, b_len)),
    }
