"""Shared text-extraction backend protocol and static backend factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from PIL import Image

from app.core.config import Settings
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox


@dataclass(frozen=True)
class OCRBackendMetadata:
    backend: str
    model_id: str
    detector_id: str | None = None
    model_revision: str | None = None
    object_metadata: Mapping[str, object] = field(default_factory=dict)


class OCRBackend(Protocol):
    """A supported full-page text extraction implementation."""

    @property
    def metadata(self) -> OCRBackendMetadata: ...

    def predict(self, image: Image.Image) -> list[OCRLine]: ...


class MockOCRBackend:
    """Deterministic development backend retained for tests and local workflows."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def metadata(self) -> OCRBackendMetadata:
        return OCRBackendMetadata(
            backend="mock",
            model_id=self.settings.ocr_model_id,
            object_metadata={"mock": True, "text_extraction_mode": "ocr"},
        )

    def predict(self, image: Image.Image) -> list[OCRLine]:
        width, height = image.size
        return [
            OCRLine(
                text="MOCK_OCR_TEXT",
                confidence=0.99,
                bbox=BBox(x1=width * 0.08, y1=height * 0.08, x2=width * 0.92, y2=height * 0.25),
            )
        ]


def _create_classic_ocr_backend(settings: Settings) -> OCRBackend:
    backend = settings.ocr_backend
    if backend == "mock":
        return MockOCRBackend(settings)
    if backend == "paddle":
        from app.modules.ocr.paddle_backend import PaddleOCRBackend

        return PaddleOCRBackend(settings)
    if backend == "bina_rizeh":
        from app.modules.ocr.bina_rizeh_backend import BinaRizehOCRBackend

        return BinaRizehOCRBackend(settings)
    raise RuntimeError(f"Unsupported OCR backend: {backend}")


def create_ocr_backend(settings: Settings) -> OCRBackend:
    """Create the configured text extractor while preserving the OCR service contract.

    ``text_extraction_mode=ocr`` selects the existing Paddle/Bina/mock path.
    ``text_extraction_mode=vlm`` selects the explicitly supported remote VLM path.
    Imports remain static so configuration cannot trigger arbitrary code imports.
    """

    if settings.text_extraction_mode == "ocr":
        return _create_classic_ocr_backend(settings)

    if settings.text_extraction_mode == "vlm":
        if settings.vlm_backend == "ollama":
            from app.modules.ocr.deepseek_ocr_vlm_backend import DeepSeekOCRVLMBackend

            return DeepSeekOCRVLMBackend(settings)
        raise RuntimeError(f"Unsupported VLM backend: {settings.vlm_backend}")

    raise RuntimeError(f"Unsupported text extraction mode: {settings.text_extraction_mode}")
