"""PP-DocLayoutV3 localization backend."""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.devices import require_paddle_device
from app.modules.figure_table.types import LayoutDetectionItem
from app.schemas.common import BBox, Point, Polygon


def _json_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        value = result
    else:
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if value is None:
            value = getattr(result, "res", None)
        if value is None:
            raise RuntimeError("Unsupported LayoutDetection result object")
    if not isinstance(value, dict):
        raise RuntimeError("Unsupported LayoutDetection result payload")
    nested = value.get("res", value)
    if not isinstance(nested, dict):
        raise RuntimeError("LayoutDetection result 'res' must be a mapping")
    return nested


def _polygon_from_box(box: list[float]) -> Polygon:
    x1, y1, x2, y2 = map(float, box)
    return Polygon(
        points=[
            Point(x=x1, y=y1),
            Point(x=x2, y=y1),
            Point(x=x2, y=y2),
            Point(x=x1, y=y2),
        ]
    )


class PPDocLayoutBackend:
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
                import paddle
                from paddleocr import LayoutDetection
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PP-DocLayout backend requested but PaddleOCR is not installed. "
                    "Install requirements-models.txt or use WIKI_HAMI_FIGURE_TABLE_BACKEND=mock."
                ) from exc
            require_paddle_device(self.settings.figure_table_device, paddle)
            self._model = LayoutDetection(
                model_name=self.settings.figure_table_model_id.split("/")[-1],
                device=self.settings.figure_table_device,
            )
        return self._model

    def predict(self, image: Image.Image) -> list[LayoutDetectionItem]:
        model = self._load()
        rgb = np.asarray(image.convert("RGB"))
        with self._predict_lock:
            results = list(model.predict(input=rgb, batch_size=1))

        items: list[LayoutDetectionItem] = []
        for result in results:
            payload = _json_payload(result)
            for box in payload.get("boxes") or []:
                if not isinstance(box, dict):
                    continue
                score = float(box.get("score", 0.0))
                if score < self.settings.figure_table_score_threshold:
                    continue
                coordinates = box.get("coordinate")
                if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                    continue
                x1, y1, x2, y2 = map(float, coordinates)
                items.append(
                    LayoutDetectionItem(
                        label=str(box.get("label", "")).strip().lower(),
                        confidence=max(0.0, min(1.0, score)),
                        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        polygon=_polygon_from_box([x1, y1, x2, y2]),
                        class_id=int(box["cls_id"]) if box.get("cls_id") is not None else None,
                    )
                )
        return items
