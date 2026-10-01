"""Bina 0.2 Rizeh full-page OCR backend.

The implementation follows the official ``bina_page_ocr.py`` wrapper: load the
bundled local detector/recognizer folders into PaddleOCR, convert visual-order
recognition to logical Persian order, and retain the original visual text.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.paddle_backend import _json_payload
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox


BINA_DETECTOR_ID = "PaddlePaddle/PP-OCRv6_medium_det"
BINA_DETECTOR_REVISION = "8e0f56fb2ef86b461d99cfc7ac5c137738985f61"
_LTR_RUN = re.compile(r"[a-zA-Z0-9 :*./%+-]")


def logical_persian_text(visual_text: str) -> str:
    """Apply Bina's published visual-to-logical line ordering routine."""
    segments: list[str] = []
    current_ltr = ""
    for character in visual_text:
        if _LTR_RUN.search(character):
            current_ltr += character
            continue
        if current_ltr:
            segments.append(current_ltr)
            current_ltr = ""
        segments.append(character)
    if current_ltr:
        segments.append(current_ltr)
    return "".join(reversed(segments))


def _number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Bina line {field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Bina line {field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"Bina line {field_name} must be finite")
    return number


def _bina_page_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact useful line schema emitted by official ``BinaPageOCR``."""
    visual_texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores")
    boxes = payload.get("rec_boxes")
    if not isinstance(visual_texts, (list, tuple)):
        raise RuntimeError("Bina PaddleOCR result rec_texts must be a sequence")
    if scores is None:
        scores = [0.0] * len(visual_texts)
    if not isinstance(scores, (list, tuple)) or len(scores) != len(visual_texts):
        raise RuntimeError("Bina PaddleOCR result rec_scores must match rec_texts")
    if not isinstance(boxes, (list, tuple, np.ndarray)) or len(boxes) != len(visual_texts):
        raise RuntimeError(
            "Bina full-page result must provide one rec_box for every recognized line"
        )

    items: list[dict[str, Any]] = []
    for index, (visual_value, box) in enumerate(zip(visual_texts, boxes, strict=True)):
        visual_text = str(visual_value)
        flattened = np.asarray(box, dtype=float).reshape(-1)
        if flattened.size != 4 or not np.isfinite(flattened).all():
            raise RuntimeError(f"Bina line {index} has an invalid rec_box")
        x0, y0, x1, y1 = (float(value) for value in flattened)
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"Bina line {index} rec_box must have positive area")
        score = _number(scores[index], field_name="score")
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Bina line {index} score must be between 0 and 1")
        items.append(
            {
                "text": logical_persian_text(visual_text),
                "score": score,
                "raw_visual_text": visual_text,
                "box": [x0, y0, x1, y1],
                "x": (x0 + x1) / 2,
                "y": (y0 + y1) / 2,
                "height": max(y1 - y0, 1.0),
            }
        )

    items.sort(key=lambda item: item["y"])
    rows: list[list[dict[str, Any]]] = []
    for item in items:
        if not rows:
            rows.append([item])
            continue
        row = rows[-1]
        mean_y = sum(part["y"] for part in row) / len(row)
        mean_height = sum(part["height"] for part in row) / len(row)
        if abs(item["y"] - mean_y) <= 0.55 * max(item["height"], mean_height):
            row.append(item)
        else:
            rows.append([item])

    lines: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda item: item["x"], reverse=True)
        for item in row:
            lines.append(
                {
                    "text": item["text"],
                    "score": item["score"],
                    "raw_visual_text": item["raw_visual_text"],
                    "box": item["box"],
                    "row": row_index,
                }
            )
    return {"lines": lines}


def parse_bina_page_output(prediction: Mapping[str, Any]) -> list[OCRLine]:
    """Parse the per-line JSON contract emitted by Bina's official page wrapper."""
    lines_value = prediction.get("lines")
    if not isinstance(lines_value, list):
        raise RuntimeError("Bina page output must contain a lines list")

    lines: list[OCRLine] = []
    for index, item in enumerate(lines_value):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Bina page output line {index} must be an object")
        text = item.get("text")
        raw_text = item.get("raw_visual_text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Bina page output line {index} must contain non-empty text")
        if not isinstance(raw_text, str):
            raise RuntimeError(f"Bina page output line {index} must contain raw_visual_text")
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise RuntimeError(f"Bina page output line {index} must contain a four-value box")
        x0, y0, x1, y1 = (_number(value, field_name="box") for value in box)
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"Bina page output line {index} box must have positive area")
        confidence = _number(item.get("score"), field_name="score")
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"Bina page output line {index} score must be between 0 and 1")
        lines.append(
            OCRLine(
                text=text,
                raw_text=raw_text,
                confidence=confidence,
                bbox=BBox(x1=x0, y1=y0, x2=x1, y2=y1),
            )
        )
    return lines


class BinaRizehOCRBackend:
    """Lazy, reusable in-process Bina full-page detector-plus-recognizer."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._init_lock = Lock()
        self._predict_lock = Lock()

    @property
    def metadata(self) -> OCRBackendMetadata:
        return OCRBackendMetadata(
            backend="bina_rizeh",
            model_id=self.settings.ocr_bina_model_id,
            detector_id=BINA_DETECTOR_ID,
            model_revision=self.settings.ocr_bina_revision,
            object_metadata={"text_detection_model_revision": BINA_DETECTOR_REVISION},
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._init_lock:
            if self._model is not None:
                return self._model
            try:
                from huggingface_hub import snapshot_download
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - depends on optional model extras
                raise RuntimeError(
                    "Bina Rizeh backend requested but its PaddleOCR and Hugging Face dependencies "
                    "are not installed. Install requirements-models.txt or use "
                    "WIKI_HAMI_OCR_BACKEND=mock."
                ) from exc
            try:
                model_root = Path(
                    snapshot_download(
                        repo_id=self.settings.ocr_bina_model_id,
                        revision=self.settings.ocr_bina_revision,
                        allow_patterns=["inference/*", "detector/*"],
                    )
                )
                recognizer_dir = model_root / "inference"
                detector_dir = model_root / "detector"
                if not recognizer_dir.is_dir() or not detector_dir.is_dir():
                    raise RuntimeError("download did not include Bina inference/ and detector/ directories")
                self._model = PaddleOCR(
                    text_detection_model_dir=str(detector_dir),
                    text_recognition_model_dir=str(recognizer_dir),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_rec_score_thresh=self.settings.ocr_bina_score_threshold,
                    device=self.settings.ocr_device,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize Bina Rizeh full-page OCR. Verify the configured "
                    "revision, persistent Hugging Face cache access, and Paddle device runtime."
                ) from exc
        return self._model

    def predict(self, image: Image.Image) -> list[OCRLine]:
        model = self._load()
        with self._predict_lock:
            results = list(model.predict(np.asarray(image.convert("RGB"))))
        if len(results) != 1:
            raise RuntimeError(
                f"Bina full-page OCR returned {len(results)} results for one prepared page; expected one"
            )
        payload = _json_payload(results[0])
        return parse_bina_page_output(_bina_page_output(payload))
