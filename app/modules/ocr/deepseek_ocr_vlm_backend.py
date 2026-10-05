"""DeepSeek-OCR VLM backend served through Ollama's REST API."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import replace
from io import BytesIO
from pathlib import Path
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
logger = logging.getLogger(__name__)


class VLMQualityError(RuntimeError):
    """Unusable model output, eligible for a bounded fresh request."""


def validate_vlm_text(text: str) -> None:
    if not text.strip() or sum(c.isalpha() for c in re.sub(r"<[^>]*>", "", text)) < 2:
        raise VLMQualityError("DeepSeek-OCR returned no usable text")
    if re.search(r"\\[A-Za-z_]+>", text):
        raise VLMQualityError("DeepSeek-OCR returned a malformed model artifact")
    if re.match(r"\Aday\s+\[\[1\]\]", text, re.I) or (
        text.count("\u200c") >= 40 and text.count("\u200c") > 2 * text.count(" ")
    ):
        raise VLMQualityError("DeepSeek-OCR returned damaged text ordering or a model header")
    if len(text) >= 960:
        span = 120
        counts = Counter(text[i:i + span] for i in range(len(text) - span + 1))
        repetitions = max(counts.values(), default=0)
        if repetitions >= 8 and repetitions * span >= len(text) * 0.2:
            raise VLMQualityError(
                f"DeepSeek-OCR contains a repeated {span}-character passage "
                f"{repetitions} times; rerun OCR with a smaller page region"
            )


def crop_document_margins(image: Image.Image) -> tuple[Image.Image, int, int]:
    """Find ink bounds after excluding long outer frame lines from the detection mask.

    The original pixels are preserved; the mask only chooses the crop boundary.
    Leave small/blank images unchanged and keep 24 pixels of padding around ink.
    """
    import numpy as np

    width, height = image.size
    mask = np.asarray(image.convert("L")) < 215
    columns = np.flatnonzero(mask.sum(axis=0) > height * 0.60)
    rows = np.flatnonzero(mask.sum(axis=1) > width * 0.60)
    for x in columns:
        if x < width * 0.08 or x > width * 0.92:
            mask[:, max(0, x - 10):min(width, x + 11)] = False
    for y in rows:
        if y < height * 0.08 or y > height * 0.92:
            mask[max(0, y - 10):min(height, y + 11), :] = False
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return image, 0, 0
    left, top = max(0, int(xs.min()) - 24), max(0, int(ys.min()) - 24)
    right, bottom = min(width, int(xs.max()) + 25), min(height, int(ys.max()) + 25)
    # Dense pages benefit from retaining the full page context. Only crop
    # sparse pages where blank margins consume at least 40% of the image.
    if (right - left < 64 or bottom - top < 64
            or (right - left) * (bottom - top) > width * height * 0.60):
        return image, 0, 0
    return image.crop((left, top, right, bottom)), left, top


def horizontal_regions(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Split a page into three disjoint top-to-bottom OCR regions.

    Cuts are selected near the one-third positions at rows with minimal dark-pixel
    density. This preserves Persian/legal document reading flow and avoids the old
    left/right column split that could slice one sentence into unrelated fragments.
    Regions cover every pixel exactly once and are always returned top to bottom.
    """
    import numpy as np

    width, height = image.size
    if height < 192:
        raise VLMQualityError("Image is too small for three-region OCR")

    gray = np.asarray(image.convert("L"))
    # Ignore a thin strip at both side edges so page borders do not dominate the
    # horizontal cut search. The source pixels themselves are never modified.
    left = min(width // 12, max(0, width // 4))
    right = max(left + 1, width - left)
    body = gray[:, left:right]
    ink = (body < 160).sum(axis=1)

    cuts = [0]
    for fraction in (1 / 3, 2 / 3):
        center = int(height * fraction)
        radius = max(1, height // 20)
        start = max(cuts[-1] + 1, center - radius)
        end = min(height - 1, center + radius)
        if end <= start:
            cut = center
        else:
            window = ink[start:end]
            minimum = window.min()
            candidates = np.flatnonzero(window == minimum) + start
            cut = int(min(candidates, key=lambda y: abs(y - center)))
        cut = min(height - 1, max(cuts[-1] + 1, cut))
        cuts.append(cut)
    cuts.append(height)

    return [(0, cuts[i], width, cuts[i + 1]) for i in range(3)]


def column_regions(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Backward-compatible alias for the top-to-bottom region splitter."""
    return horizontal_regions(image)


def translate_lines(lines: list[OCRLine], x: int, y: int) -> list[OCRLine]:
    return [replace(
        line,
        bbox=BBox(x1=line.bbox.x1 + x, y1=line.bbox.y1 + y,
                  x2=line.bbox.x2 + x, y2=line.bbox.y2 + y),
        polygon=Polygon(points=[Point(x=p.x + x, y=p.y + y)
                                for p in line.polygon.points]) if line.polygon else None,
    ) for line in lines]


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
    # Some model templates emit a malformed token/header block before a Markdown rule.
    cleaned = re.sub(r"\A\s*\\?<[/\\|]*im_?start>[^\u0600-\u06ff]{0,256}?\s---(?:\s|$)", "", cleaned, flags=re.I | re.S)

    # Strip wrapper junk at the beginning, including cases observed in
    # production such as ``*<|im_end|> text``.
    for _ in range(4):
        updated = _LEADING_MODEL_ARTIFACT_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated.lstrip()

    # Remove any remaining control tokens without touching OCR words around them.
    cleaned = re.sub(r"\\?<[/\\|]*im_?(?:start|end)>|<\\?\|[^>]*>", "", cleaned, flags=re.I)
    cleaned = _MODEL_CONTROL_TOKEN_RE.sub("", cleaned)
    cleaned = _HTML_BREAK_RE.sub("\n", cleaned)

    # Canonical artifacts are plain text rather than Markdown. Preserve list
    # semantics using '-' while dropping presentation-only markup.
    cleaned = _MARKDOWN_FENCE_RE.sub("", cleaned)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _MARKDOWN_UNORDERED_LIST_RE.sub(r"\1- ", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = _MARKDOWN_ORPHAN_MARKER_RE.sub("", cleaned)
    cleaned = re.sub(r"(?m)^\s*\\[*_`]+\s*$", "", cleaned)
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
        raise VLMQualityError("DeepSeek-OCR returned empty content")

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
    validate_vlm_text("\n".join(line.text for line in lines))
    ordered = sorted(lines, key=lambda line: (line.bbox.y1, -line.bbox.x1))
    return [replace(line, reading_order=index) for index, line in enumerate(ordered)]


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
        raise VLMQualityError("DeepSeek-OCR returned empty content")

    if _GROUNDING_PAIR_RE.search(content):
        return parse_deepseek_grounding_output(
            content,
            image_width=image_width,
            image_height=image_height,
        )

    raw_text = _clean_plain_text_response(content)
    text = normalize_persian_ocr_text(raw_text)
    validate_vlm_text(text)

    return [
        OCRLine(
            text=text,
            raw_text=raw_text,
            confidence=0.0,
            bbox=BBox(x1=0.0, y1=0.0, x2=float(image_width), y2=float(image_height)),
            polygon=None,
            reading_order=0,
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

    def _request(self, image: Image.Image, *, prompt: str | None = None) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        url = f"{self.settings.vlm_base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.settings.vlm_model_id,
            "messages": [
                {
                    "role": "user",
                    "content": self.settings.vlm_prompt if prompt is None else prompt,
                    "images": [encoded],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.settings.vlm_max_tokens,
                "num_ctx": self.settings.vlm_context_size,
                "repeat_penalty": self.settings.vlm_repeat_penalty,
            },
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

        if self.settings.vlm_diagnostics_dir:
            # Deterministic per-image/request filenames bound repeated retry storage.
            digest = hashlib.sha256(buffer.getvalue())
            digest.update(json.dumps({k: v for k, v in payload.items() if k != "messages"}, sort_keys=True).encode())
            digest.update(payload["messages"][0]["content"].encode())
            try:
                directory = Path(self.settings.vlm_diagnostics_dir)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{digest.hexdigest()}.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8",
                )
            except OSError:
                logger.exception("Could not retain local OCR response diagnostics")
        if not isinstance(body, dict):
            raise RuntimeError("DeepSeek-OCR Ollama response must be a JSON object")
        if body.get("error"):
            raise RuntimeError(f"DeepSeek-OCR Ollama error: {body['error']}")
        if body.get("done") is not True or body.get("done_reason") == "length":
            raise VLMQualityError(
                "DeepSeek-OCR generation is incomplete or hit its token limit; "
                "rerun OCR with a smaller page region"
            )
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek-OCR Ollama response is missing message.content")
        return content

    def predict(self, image: Image.Image) -> list[OCRLine]:
        offset_x = offset_y = 0
        if self.settings.vlm_crop_margins:
            image, offset_x, offset_y = crop_document_margins(image)
        width, height = image.size
        for attempt in range(self.settings.vlm_quality_retries + 1):
            try:
                # No chat history; retry using the documented alternative OCR task.
                prompt = None if attempt == 0 else "\nFree OCR."
                content = self._request(image, prompt=prompt)
                lines = parse_deepseek_output(
                    content, image_width=width, image_height=height,
                )
                if offset_x or offset_y:
                    lines = translate_lines(lines, offset_x, offset_y)
                return lines
            except VLMQualityError as exc:
                if attempt == self.settings.vlm_quality_retries:
                    raise
                logger.warning("Rejected VLM OCR; retrying fresh image request: %s", exc)
        raise AssertionError("unreachable")

    def predict_regions(self, image: Image.Image) -> list[OCRLine]:
        """Recover a difficult page using three independent top-to-bottom regions."""
        offset_x = offset_y = 0
        if self.settings.vlm_crop_margins:
            image, offset_x, offset_y = crop_document_margins(image)

        output: list[OCRLine] = []
        for box in horizontal_regions(image):
            region = image.crop(box)
            content = self._request(region, prompt="\nExtract the text in the image.")
            lines = parse_deepseek_output(
                content, image_width=region.width, image_height=region.height,
            )
            translated = translate_lines(
                lines,
                box[0] + offset_x,
                box[1] + offset_y,
            )
            for line in translated:
                output.append(replace(line, reading_order=len(output)))
        validate_vlm_text("\n".join(line.text for line in output))
        return output
