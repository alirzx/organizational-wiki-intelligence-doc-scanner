"""PaddleOCR backend for page-level text detection and recognition.

The backend deliberately returns model-space OCRLine objects. Mapping to the Wiki Hami
canonical schema and source-image coordinates is handled by adapter.py.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox, Point, Polygon


def _json_payload(result: Any) -> dict[str, Any]:
    """Normalize PaddleOCR Result/dict variants into the inner ``res`` payload."""
    if isinstance(result, dict):
        value = result
    else:
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if value is None:
            value = getattr(result, "res", None)
        if value is None:
            raise RuntimeError("Unsupported PaddleOCR result object: missing json/res payload")
    if not isinstance(value, dict):
        raise RuntimeError("Unsupported PaddleOCR result payload")
    nested = value.get("res", value)
    if not isinstance(nested, dict):
        raise RuntimeError("PaddleOCR result 'res' must be a mapping")
    return nested


def _polygon_from_points(points: Any) -> Polygon | None:
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return None
    return Polygon(points=[Point(x=float(x), y=float(y)) for x, y in arr.tolist()])


def _bbox_from_box_or_polygon(box: Any, polygon: Any) -> BBox:
    if box is not None:
        arr = np.asarray(box, dtype=float).reshape(-1)
        if arr.size == 4:
            return BBox(x1=float(arr[0]), y1=float(arr[1]), x2=float(arr[2]), y2=float(arr[3]))

    arr = np.asarray(polygon, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError("PaddleOCR result contains neither a valid rec_box nor rec_polygon")
    xs = arr[:, 0]
    ys = arr[:, 1]
    return BBox(x1=float(xs.min()), y1=float(ys.min()), x2=float(xs.max()), y2=float(ys.max()))


class PaddleOCRBackend:
    """Lazy, reusable PaddleOCR pipeline.

    A lock protects one model instance because the same backend is shared by all page jobs.
    Different Wiki Hami modules still execute concurrently in separate worker threads.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._init_lock = Lock()
        self._predict_lock = Lock()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._init_lock:
            if self._model is not None:
                return self._model
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - only when model extra is absent
                raise RuntimeError(
                    "PaddleOCR backend requested but paddleocr/paddlepaddle are not installed. "
                    "Install requirements-models.txt or use WIKI_HAMI_OCR_BACKEND=mock."
                ) from exc

            device = self.settings.ocr_device
            self._model = PaddleOCR(
                text_detection_model_name=self.settings.ocr_text_detection_model_name,
                text_recognition_model_name=self.settings.ocr_model_id.split("/")[-1],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self.settings.ocr_use_textline_orientation,
                device=device,
                enable_mkldnn=self.settings.ocr_enable_mkldnn,
            )
        return self._model

    def predict(self, image: Image.Image) -> list[OCRLine]:
        model = self._load()
        rgb = np.asarray(image.convert("RGB"))
        with self._predict_lock:
            results = list(model.predict(rgb))

        lines: list[OCRLine] = []
        for result in results:
            payload = _json_payload(result)
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            boxes = payload.get("rec_boxes")
            polygons = payload.get("rec_polys") or payload.get("dt_polys") or []

            boxes_list = list(np.asarray(boxes).tolist()) if boxes is not None else [None] * len(texts)
            polygons_list = list(polygons) if polygons is not None else []

            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else 0.0
                if score < self.settings.ocr_score_threshold or not str(text).strip():
                    continue
                polygon_raw = polygons_list[index] if index < len(polygons_list) else None
                box_raw = boxes_list[index] if index < len(boxes_list) else None
                bbox = _bbox_from_box_or_polygon(box_raw, polygon_raw)
                polygon = _polygon_from_points(polygon_raw)
                lines.append(
                    OCRLine(
                        text=str(text),
                        confidence=max(0.0, min(1.0, score)),
                        bbox=bbox,
                        polygon=polygon,
                    )
                )

        return sorted(lines, key=lambda line: (line.bbox.y1, line.bbox.x1))
