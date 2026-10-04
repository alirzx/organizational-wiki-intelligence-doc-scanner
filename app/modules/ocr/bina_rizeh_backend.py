"""Full-page OCR using PP-OCRv6 detection and Bina line recognition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.devices import require_paddle_device
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.paddle_backend import _json_payload
from app.modules.ocr.text_normalization import (
    normalize_persian_ocr_text,
    visual_persian_to_logical,
)
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox, Point, Polygon


BINA_RUNTIME_FILES = (
    "inference/inference.json",
    "inference/inference.pdiparams",
    "inference/inference.yml",
)
DETECTOR_RUNTIME_FILES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class _DetectedLine:
    bbox: BBox
    polygon: Polygon
    points: np.ndarray
    center_x: float
    center_y: float
    height: float


def logical_persian_text(visual_text: str) -> str:
    """Convert Bina visual-order output to normalized logical Persian text."""

    return normalize_persian_ocr_text(visual_persian_to_logical(visual_text))


def _number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Bina {field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Bina {field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"Bina {field_name} must be finite")
    return number


def _validate_runtime_files(root: Path, required: Sequence[str], *, label: str) -> None:
    for relative_name in required:
        path = root / relative_name
        if not path.is_file():
            raise RuntimeError(f"{label} runtime artifact is missing: {relative_name}")
        with path.open("rb") as artifact:
            prefix = artifact.read(128)
        if prefix.startswith(_LFS_POINTER):
            raise RuntimeError(f"{label} runtime artifact is a Git LFS pointer: {relative_name}")
        if path.suffix == ".pdiparams" and path.stat().st_size < 1024:
            raise RuntimeError(f"{label} runtime weights are unexpectedly small: {relative_name}")


def _single_payload(results: Any, *, stage: str) -> Mapping[str, Any]:
    values = list(results)
    if len(values) != 1:
        raise RuntimeError(f"Bina {stage} returned {len(values)} results for one input; expected one")
    return _json_payload(values[0])


def parse_bina_detection_output(
    prediction: Mapping[str, Any], *, image_width: int, image_height: int
) -> list[_DetectedLine]:
    """Validate detector polygons in prepared-page pixel coordinates."""
    polygons_value = prediction.get("dt_polys")
    if polygons_value is None:
        raise RuntimeError("Bina detector output must contain dt_polys")
    if not isinstance(polygons_value, (list, tuple, np.ndarray)):
        raise RuntimeError("Bina detector dt_polys must be a sequence")

    detections: list[_DetectedLine] = []
    for index, value in enumerate(polygons_value):
        points = np.asarray(value, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise RuntimeError(f"Bina detector polygon {index} must contain four finite points")
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        x0 = float(points[:, 0].min())
        y0 = float(points[:, 1].min())
        x1 = float(points[:, 0].max())
        y1 = float(points[:, 1].max())
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"Bina detector polygon {index} has no area after clipping")
        polygon = Polygon(
            points=[Point(x=float(point[0]), y=float(point[1])) for point in points]
        )
        detections.append(
            _DetectedLine(
                bbox=BBox(x1=x0, y1=y0, x2=x1, y2=y1),
                polygon=polygon,
                points=points.astype(np.float32),
                center_x=(x0 + x1) / 2,
                center_y=(y0 + y1) / 2,
                height=y1 - y0,
            )
        )
    return detections


def _sort_reading_order(detections: list[_DetectedLine]) -> list[_DetectedLine]:
    pending = sorted(detections, key=lambda item: (item.center_y, -item.center_x))
    rows: list[list[_DetectedLine]] = []
    for detection in pending:
        if not rows:
            rows.append([detection])
            continue
        row = rows[-1]
        mean_y = sum(item.center_y for item in row) / len(row)
        mean_height = sum(item.height for item in row) / len(row)
        if abs(detection.center_y - mean_y) <= 0.55 * max(detection.height, mean_height):
            row.append(detection)
        else:
            rows.append([detection])
    return [item for row in rows for item in sorted(row, key=lambda item: -item.center_x)]


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered = np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    if len(np.unique(ordered, axis=0)) != 4:
        raise RuntimeError("Bina detector polygon cannot be rectified into four corners")
    return ordered


def _rectified_crop(image_bgr: np.ndarray, detection: _DetectedLine, cv2: Any) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _ordered_quad(detection.points)
    width = max(
        int(round(np.linalg.norm(top_right - top_left))),
        int(round(np.linalg.norm(bottom_right - bottom_left))),
    )
    height = max(
        int(round(np.linalg.norm(bottom_left - top_left))),
        int(round(np.linalg.norm(bottom_right - top_right))),
    )
    if width < 2 or height < 2:
        raise RuntimeError("Bina detector polygon produced an empty rectified crop")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    crop = cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1] * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return np.ascontiguousarray(crop)


def parse_bina_recognition_output(prediction: Mapping[str, Any], *, index: int) -> tuple[str, float]:
    """Parse one official TextRecognition result without fabricating defaults."""
    raw_text = prediction.get("rec_text")
    if not isinstance(raw_text, str):
        raise RuntimeError(f"Bina recognition result {index} must contain rec_text")
    if "rec_score" not in prediction:
        raise RuntimeError(f"Bina recognition result {index} must contain rec_score")
    score = _number(prediction.get("rec_score"), field_name=f"recognition result {index} score")
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Bina recognition result {index} score must be between 0 and 1")
    return raw_text, score


class BinaRizehOCRBackend:
    """Lazy, reusable detector-plus-Bina-recognizer full-page backend."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._detector: Any | None = None
        self._recognizer: Any | None = None
        self._cv2: Any | None = None
        self._init_lock = Lock()
        self._predict_lock = Lock()

    @property
    def metadata(self) -> OCRBackendMetadata:
        return OCRBackendMetadata(
            backend="bina_rizeh",
            model_id=self.settings.ocr_bina_model_id,
            detector_id=self.settings.ocr_detection_model_id,
            model_revision=self.settings.ocr_bina_revision,
            object_metadata={
                "text_detection_model_revision": self.settings.ocr_detection_model_revision
            },
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._detector is not None and self._recognizer is not None and self._cv2 is not None:
            return self._detector, self._recognizer, self._cv2
        with self._init_lock:
            if self._detector is not None and self._recognizer is not None and self._cv2 is not None:
                return self._detector, self._recognizer, self._cv2
            try:
                import cv2
                import paddle
                from huggingface_hub import snapshot_download
                from paddleocr import TextDetection, TextRecognition
            except ImportError as exc:  # pragma: no cover - optional model extras
                raise RuntimeError(
                    "Bina Rizeh backend requires PaddleOCR, PaddlePaddle, OpenCV, and "
                    "huggingface_hub. Install the configured model runtime dependencies."
                ) from exc

            require_paddle_device(self.settings.ocr_device, paddle)
            try:
                recognizer_root = Path(
                    snapshot_download(
                        repo_id=self.settings.ocr_bina_model_id,
                        revision=self.settings.ocr_bina_revision,
                        allow_patterns=list(BINA_RUNTIME_FILES),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download pinned Bina recognition runtime artifacts. Check the "
                    "model ID, revision, Hugging Face access, and persistent cache permissions."
                ) from exc
            try:
                detector_root = Path(
                    snapshot_download(
                        repo_id=self.settings.ocr_detection_model_id,
                        revision=self.settings.ocr_detection_model_revision,
                        allow_patterns=list(DETECTOR_RUNTIME_FILES),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download pinned OCR detector runtime artifacts. Check the model "
                    "ID, revision, Hugging Face access, and persistent cache permissions."
                ) from exc

            _validate_runtime_files(recognizer_root, BINA_RUNTIME_FILES, label="Bina recognizer")
            _validate_runtime_files(detector_root, DETECTOR_RUNTIME_FILES, label="OCR detector")
            try:
                detector = TextDetection(
                    model_dir=str(detector_root),
                    device=self.settings.ocr_device,
                    enable_mkldnn=self.settings.ocr_enable_mkldnn,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the PP-OCRv6 text detector from validated artifacts"
                ) from exc
            try:
                recognizer = TextRecognition(
                    model_dir=str(recognizer_root / "inference"),
                    device=self.settings.ocr_device,
                    enable_mkldnn=self.settings.ocr_enable_mkldnn,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize the Bina line recognizer from validated artifacts"
                ) from exc
            self._detector = detector
            self._recognizer = recognizer
            self._cv2 = cv2
        return self._detector, self._recognizer, self._cv2

    def predict(self, image: Image.Image) -> list[OCRLine]:
        detector, recognizer, cv2 = self._load()
        rgb = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._predict_lock:
            try:
                detection_payload = _single_payload(
                    detector.predict(input=image_bgr, batch_size=1), stage="detector"
                )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith("Bina detector"):
                    raise
                raise RuntimeError("Bina text detection inference failed") from exc
            detections = _sort_reading_order(
                parse_bina_detection_output(
                    detection_payload,
                    image_width=image_bgr.shape[1],
                    image_height=image_bgr.shape[0],
                )
            )
            if not detections:
                return []
            crops = [_rectified_crop(image_bgr, detection, cv2) for detection in detections]
            if self.settings.ocr_bina_batch_size < 1:
                raise RuntimeError("WIKI_HAMI_OCR_BINA_BATCH_SIZE must be at least 1")
            try:
                recognition_results = list(
                    recognizer.predict(
                        input=crops,
                        batch_size=self.settings.ocr_bina_batch_size,
                    )
                )
            except Exception as exc:
                raise RuntimeError("Bina text recognition inference failed") from exc

        if len(recognition_results) != len(detections):
            raise RuntimeError(
                "Bina recognizer returned a different number of results than detector crops"
            )
        lines: list[OCRLine] = []
        for index, (result, detection) in enumerate(
            zip(recognition_results, detections, strict=True)
        ):
            raw_text, confidence = parse_bina_recognition_output(
                _json_payload(result), index=index
            )
            text = logical_persian_text(raw_text)
            if confidence < self.settings.ocr_bina_score_threshold or not text.strip():
                continue
            lines.append(
                OCRLine(
                    text=text,
                    raw_text=raw_text,
                    confidence=confidence,
                    bbox=detection.bbox,
                    polygon=detection.polygon,
                )
            )
        return lines
