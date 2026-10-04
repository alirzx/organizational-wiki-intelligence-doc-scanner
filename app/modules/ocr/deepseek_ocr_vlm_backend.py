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
_MODEL_CONTROL_TOKEN_RE = re.compile(
    r"(?:\\?<\|(?:im_start|im_end)\|>|\\?</?im_(?:start|end)>|<\\?/?im_(?:start|end)>)",
    re.IGNORECASE,
)
_LEADING_MODEL_ARTIFACT_RE = re.compile(
    r"^\s*[*_`#-]*\s*(?:\\?<\|(?:im_start|im_end)\|>|\\?</?im_(?:start|end)>|<\\?/?im_(?:start|end)>)(?:\s*<br\s*/?>)?\s*",
    re.IGNORECASE,
)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_MARKDOWN_FENCE_RE = re.compile(r"(?m)^\s*```[^\n]*\s*$")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}[ \t]+")
_MARKDOWN_UNORDERED_LIST_RE = re.compile(r"(?m)^(\s*)[*+][ \t]+")
_MARKDOWN_ORPHAN_MARKER_RE = re.compile(r"(?m)^\s*[*_`]{1,3}\s*$")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
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


def _clean_plain_text_response(content: str) -> str:
    """Remove model/template artifacts while preserving OCR lexical content.

    Some Ollama DeepSeek-OCR templates leak image-boundary tokens in either the
    XML-like form (``</im_start>``) or the tokenizer form (``<|im_end|>``). The
    production frontend must never receive those transport artifacts. Markdown
    decoration emitted by document-mode OCR is reduced to plain-text structure,
    while actual recognized words, punctuation, list order, and line breaks are
    retained.
    """

    cleaned = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    # Strip wrapper junk at the beginning, including cases observed in
    # production such as ``*<|im_end|> text``.
    for _ in range(4):
        updated = _LEADING_MODEL_ARTIFACT_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated.lstrip()

    # Remove any remaining control tokens without touching OCR words around them.
    cleaned = _MODEL_CONTROL_TOKEN_RE.sub("", cleaned)
    cleaned = _HTML_BREAK_RE.sub("\n", cleaned)

    # Canonical artifacts are plain text rather than Markdown. Preserve list
    # semantics using '-' while dropping presentation-only markup.
    cleaned = _MARKDOWN_FENCE_RE.sub("", cleaned)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _MARKDOWN_UNORDERED_LIST_RE.sub(r"\1- ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = _MARKDOWN_ORPHAN_MARKER_RE.sub("", cleaned)
    cleaned = _EXCESS_BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


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
        raise RuntimeError("DeepSeek-OCR response did not contain grounding tags")

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
        lines.append(
            OCRLine(
                text=text,
                raw_text=raw_text,
                confidence=0.0,
                bbox=bbox,
                polygon=polygon,
            )
        )

    if not lines:
        raise RuntimeError("DeepSeek-OCR grounding response contained no text spans")
    return sorted(lines, key=lambda line: (line.bbox.y1, -line.bbox.x1))


def parse_deepseek_output(
    content: str,
    *,
    image_width: int,
    image_height: int,
) -> list[OCRLine]:
    """Parse grounded output when available, otherwise preserve full-page OCR text.

    The deployed Ollama build may return document text/Markdown without ``ref``/
    ``det`` tags. In that case the V1 object schema stays unchanged: one full-page
    text object is returned, while metadata marks geometry/confidence unavailable.
    """

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek-OCR returned empty content")

    if _GROUNDING_PAIR_RE.search(content):
        return parse_deepseek_grounding_output(
            content,
            image_width=image_width,
            image_height=image_height,
        )

    raw_text = _clean_plain_text_response(content)
    text = normalize_persian_ocr_text(raw_text)
    if not text:
        raise RuntimeError("DeepSeek-OCR returned no usable text")

    return [
        OCRLine(
            text=text,
            raw_text=raw_text,
            confidence=0.0,
            bbox=BBox(x1=0.0, y1=0.0, x2=float(image_width), y2=float(image_height)),
            polygon=None,
        )
    ]


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
                "grounding_coordinate_space": "deepseek_0_999_when_available",
                "geometry_available": "response_dependent",
                "confidence_available": False,
                "confidence_semantics": "unavailable_sentinel_zero",
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
        return parse_deepseek_output(
            content,
            image_width=width,
            image_height=height,
        )
