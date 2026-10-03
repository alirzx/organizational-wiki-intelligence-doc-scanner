"""RF-DETR backend for stamp/signature localization.

Baseline checkpoint:
bluecopa/rf-detr-stamp-signature-detector

The upstream checkpoint also contains checkbox classes. Wiki Hami V1 intentionally keeps
only stamp and signature.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.devices import require_torch_device
from app.modules.stamp_signature.types import MarkDetection
from app.schemas.common import BBox


_CLASS_NAMES = {
    0: "stamp",
    1: "signature",
    2: "checkbox_checked",
    3: "checkbox_unchecked",
}


class RFDETRStampSignatureBackend:
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
                import torch
                from huggingface_hub import hf_hub_download
                from rfdetr import RFDETRBase
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "RF-DETR backend requested but rfdetr/huggingface_hub are not installed. "
                    "Install requirements-models.txt or use WIKI_HAMI_STAMP_SIGNATURE_BACKEND=mock."
                ) from exc

            require_torch_device(self.settings.stamp_signature_device, torch)
            checkpoint = hf_hub_download(
                repo_id=self.settings.stamp_signature_model_id,
                filename=self.settings.stamp_signature_checkpoint_filename,
                revision=self.settings.stamp_signature_model_revision,
                cache_dir=self.settings.stamp_signature_cache_dir,
            )

            # Current RF-DETR recommends passing fine-tuned checkpoints as pretrain_weights.
            # Keep a compatibility fallback for older versions/model-card examples exposing .load().
            try:
                model = RFDETRBase(pretrain_weights=checkpoint, num_classes=4)
            except (TypeError, AttributeError):
                model = RFDETRBase()
                loader = getattr(model, "load", None)
                if loader is None:
                    raise RuntimeError("Installed RF-DETR version cannot load the selected checkpoint")
                loader(checkpoint)

            requested_device = self.settings.stamp_signature_device.strip().lower()
            if requested_device and requested_device != "auto":
                target = torch.device(requested_device)
                context = getattr(model, "model", None)
                module = getattr(context, "model", None)
                if context is None or module is None:
                    raise RuntimeError("RF-DETR model context does not expose a movable torch module")
                context.model = module.to(target)
                context.device = target

            self._model = model
        return self._model

    @staticmethod
    def _from_supervision_detections(detections: Any) -> list[MarkDetection]:
        xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=float)
        confidences = np.asarray(getattr(detections, "confidence", []), dtype=float)
        class_ids = np.asarray(getattr(detections, "class_id", []), dtype=int)
        data = getattr(detections, "data", {}) or {}
        class_names = data.get("class_name") if isinstance(data, dict) else None
        if class_names is not None:
            class_names = np.asarray(class_names, dtype=object)

        items: list[MarkDetection] = []
        for index, box in enumerate(xyxy):
            class_id = int(class_ids[index]) if index < len(class_ids) else None
            provided_name = (
                str(class_names[index])
                if class_names is not None and index < len(class_names) and str(class_names[index])
                else ""
            )
            # This Wiki Hami backend is intentionally checkpoint-specific; its published class IDs
            # are stable and safer than relying on generic COCO names from a mismatched runtime.
            label = _CLASS_NAMES.get(class_id, provided_name)
            if label not in {"stamp", "signature"}:
                continue
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            if confidence < 0.0:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            items.append(
                MarkDetection(
                    label=label,
                    confidence=max(0.0, min(1.0, confidence)),
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=class_id,
                )
            )
        return items

    @staticmethod
    def _from_legacy_iterable(detections: Any) -> list[MarkDetection]:
        items: list[MarkDetection] = []
        try:
            iterator = iter(detections)
        except TypeError:
            return items
        for detection in iterator:
            if not isinstance(detection, dict):
                continue
            raw_class = detection.get("class", detection.get("class_id"))
            class_id = int(raw_class) if isinstance(raw_class, (int, np.integer)) else None
            label = str(detection.get("class_name") or _CLASS_NAMES.get(class_id, raw_class or "")).lower()
            if label not in {"stamp", "signature"}:
                continue
            confidence = float(detection.get("confidence", detection.get("score", 0.0)))
            box = detection.get("bbox", detection.get("xyxy"))
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            items.append(
                MarkDetection(
                    label=label,
                    confidence=max(0.0, min(1.0, confidence)),
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=class_id,
                )
            )
        return items

    def predict(self, image: Image.Image) -> list[MarkDetection]:
        model = self._load()
        with self._predict_lock:
            detections = model.predict(
                image.convert("RGB"),
                threshold=self.settings.stamp_signature_score_threshold,
                include_source_image=False,
            )

        if hasattr(detections, "xyxy"):
            return self._from_supervision_detections(detections)
        return self._from_legacy_iterable(detections)
