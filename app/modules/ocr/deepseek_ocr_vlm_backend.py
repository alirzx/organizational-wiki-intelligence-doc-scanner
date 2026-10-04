"""DeepSeek-OCR VLM backend served through Ollama's REST API."""

from __future__ import annotations

import ast
import base64
import math
import re
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from app.core.config import Settings
from app.modules.ocr.backend import OCRBackendMetadata
from app.modules.ocr.text_normalization import normalize_persian_ocr_text
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox, Point, Polygon


_GROUNDING_PAIR_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)
_DEEPSEEK_COORD_MAX = 999.0


def _parse_coordinate_boxes(value: str, *, index: int) -> list[list[float]]:
    try:
        parsed = ast.literal_eval(value.strip())
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(f"DeepSeek-OCR grounding item {index} has invalid coordinates") from exc

    if isinstance(parsed, (list, tuple)) and len(parsed) == 4 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in parsed
    ):
        parsed = [parsed]

    if not isinstance(parsed, (list, tuple)):
        raise RuntimeError(f"DeepSeek-OCR grounding item {index} coordinates must be a list")

    boxes: list[list[float]] = []
    for box in parsed:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise RuntimeError(
                f"DeepSeek-OCR grounding item {index} must contain 4-value boxes"
            )
        numeric = [float(item) for item in box]
        if not all(math.isfinite(item) for item in numeric):
            raise RuntimeError(
                f"DeepSeek-OCR grounding item {index} coordinates must be finite"
            )
        x1, y1, x2, y2 = [
            min(_DEEPSEEK_COORD_MAX, max(0.0, item)) for item in numeric
        ]
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                f"DeepSeek-OCR grounding item {index} contains an empty box"
            )
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        raise RuntimeError(f"DeepSeek-OCR grounding item {index} contains no boxes")
    return boxes


def parse_deepseek_grounding_output(
    content: str,
    *,
    image_width: int,
    image_height: int,
) -> list[OCRLine]:
    """Convert DeepSeek-OCR 0..999 grounding boxes into prepared-image OCR lines."""

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek-OCR returned empty content")

    matches = list(_GROUNDING_PAIR_RE.finditer(content))
    if not matches:
        raise RuntimeError(
            "DeepSeek-OCR response did not contain grounding tags; keep the configured "
            "prompt in grounding OCR mode"
        )

    lines: list[OCRLine] = []
    for index, match in enumerate(matches):
        raw_text = match.group(1).strip()
        text = normalize_persian_ocr_text(raw_text)
        if not text:
            continue

        boxes = _parse_coordinate_boxes(match.group(2), index=index)
        x1 = min(box[0] for box in boxes) / _DEEPSEEK_COORD_MAX * image_width
        y1 = min(box[1] for box in boxes) / _DEEPSEEK_COORD_MAX * image_height
        x2 = max(box[2] for box in boxes) / _DEEPSEEK_COORD_MAX * image_width
        y2 = max(box[3] for box in boxes) / _DEEPSEEK_COORD_MAX * image_height
        bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        polygon = Polygon(
            points=[
                Point(x=x1, y=y1),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
                Point(x=x1, y=y2),
            ]
        )
        # Ollama/DeepSeek-OCR does not expose a per-span confidence score. Keep
        # the required V1 schema numeric field stable and declare its semantics
        # explicitly in backend metadata rather than inventing a probability.
        lines.append(
            OCRLine(
                text=text,
                raw_text=raw_text,
                confidence=1.0,
                bbox=bbox,
                polygon=polygon,
            )
        )

    if not lines:
        raise RuntimeError("DeepSeek-OCR grounding response contained no text spans")
    return sorted(lines, key=lambda line: (line.bbox.y1, -line.bbox.x1))


class DeepSeekOCRVLMBackend:
    """Remote DeepSeek-OCR vision-language backend using Ollama /api/chat."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def metadata(self) -> OCRBackendMetadata:
        return OCRBackendMetadata(
            backend="ollama_vlm",
            model_id=self.settings.vlm_model_id,
            object_metadata={
                "text_extraction_mode": "vlm",
                "vlm_backend": self.settings.vlm_backend,
                "grounding_coordinate_space": "deepseek_0_999",
                "confidence_available": False,
                "confidence_semantics": "schema_placeholder_only",
            },
        )

    def _request(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        url = f"{self.settings.vlm_base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.settings.vlm_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": self.settings.vlm_prompt,
                    "images": [encoded],
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(10.0, self.settings.vlm_timeout_seconds),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"DeepSeek-OCR Ollama request failed for {self.settings.vlm_base_url}"
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("DeepSeek-OCR Ollama response was not valid JSON") from exc

        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek-OCR Ollama response is missing message.content")
        return content

    def predict(self, image: Image.Image) -> list[OCRLine]:
        content = self._request(image)
        width, height = image.size
        return parse_deepseek_grounding_output(
            content,
            image_width=width,
            image_height=height,
        )
