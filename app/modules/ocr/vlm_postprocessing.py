"""Canonical cleanup helpers for VLM OCR text.

The native Ollama response is retained separately in OCR diagnostics. Text that enters
canonical OCR artifacts must not contain model transport/control tokens, chat roles,
tool metadata, embedded image payloads, or model-generated image descriptions.
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.modules.ocr.text_normalization import normalize_persian_ocr_text
from app.modules.ocr.types import OCRLine


VLM_ARTIFACT_DICTIONARY: dict[str, tuple[str, ...]] = {
    "image_transport": (
        "im_start",
        "im_end",
        "im_continue",
        "im_begin_block",
        "im_end_block",
        "image_start",
        "image_end",
        "image_pad",
        "vision_start",
        "vision_end",
    ),
    "roles": (
        "assistant",
        "user",
        "system",
        "client",
        "tool",
        "developer",
        "human",
        "bot",
    ),
    "grounding": (
        "ref",
        "det",
        "p",
        "document",
    ),
    "sequence": (
        "bos",
        "eos",
        "pad",
        "unk",
        "endoftext",
        "eot_id",
        "start_header_id",
        "end_header_id",
    ),
    "legacy_pipe": (
        "m",
        "m_end",
        "c",
    ),
    "presentation_tags": (
        "p",
        "div",
        "span",
        "section",
        "article",
        "br",
        "ol",
        "ul",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
    ),
}

_ROLE_ALIASES = (
    *VLM_ARTIFACT_DICTIONARY["roles"],
    "用户",
    "助手",
    "系统",
)

_CONTROL_TOKEN_NAMES = tuple(
    sorted(
        {
            token
            for group in (
                "image_transport",
                "roles",
                "grounding",
                "sequence",
                "legacy_pipe",
            )
            for token in VLM_ARTIFACT_DICTIONARY[group]
        },
        key=len,
        reverse=True,
    )
)
_CONTROL_TOKEN_ALT = "|".join(re.escape(token) for token in _CONTROL_TOKEN_NAMES)
_ROLE_ALT = "|".join(re.escape(token) for token in _ROLE_ALIASES)
_PRESENTATION_TAG_ALT = "|".join(
    re.escape(token) for token in VLM_ARTIFACT_DICTIONARY["presentation_tags"]
)

# Generic angle-pipe transport tokens. The closing ``>`` is optional because real
# model output has included malformed fragments such as ``<|im_continue|``.
_ANGLE_PIPE_TOKEN_RE = re.compile(
    r"<\s*/?\s*[|｜]\s*[^<>\n|｜]{1,80}\s*[|｜](?:\s*>)?",
    re.IGNORECASE,
)
# Escaped/bare variants observed through Ollama templates, e.g. ``\ |im_end|>``.
_BARE_PIPE_TOKEN_RE = re.compile(
    rf"\\?\s*[|｜]\s*(?:{_CONTROL_TOKEN_ALT})\s*[|｜]\s*>?",
    re.IGNORECASE,
)
# Role wrappers arrive in several malformed forms: </assistant>, </|assistant>,
# <|assistant|>, <|user>, etc. Accept optional pipes on both sides of the role name.
_XML_ROLE_TOKEN_RE = re.compile(
    rf"<\s*/?\s*[|｜]?\s*(?:{_ROLE_ALT}|pdfmachine)\s*[|｜]?\s*>",
    re.IGNORECASE,
)
_STANDALONE_ROLE_RE = re.compile(
    rf"^\s*(?:{_ROLE_ALT})\s*:?\s*$",
    re.IGNORECASE,
)
_ROLE_PREFIX_RE = re.compile(
    rf"^\s*(?:{_ROLE_ALT})\s*:\s*",
    re.IGNORECASE,
)
_META_MARKER_RE = re.compile(
    r"^\s*\[(?:document|question|answer|image|caption)\]\s*$",
    re.IGNORECASE,
)
_PDF_TOOL_RE = re.compile(
    r'''^\s*function\s*=\s*["']get_pdf_[^"']+["']\s*/?>?\s*$''',
    re.IGNORECASE,
)
_DATA_URI_RE = re.compile(
    r"(?i)(?:\b(?:assistant|user)\s*:\s*)?"
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/= \t]+"
)
_GROUNDING_IMAGE_RE = re.compile(
    r"^\s*image\s*\[\[\s*[-0-9.,\s]+\]\]\s*$",
    re.IGNORECASE,
)
_PRESENTATION_TAG_RE = re.compile(
    rf"</?\s*(?:{_PRESENTATION_TAG_ALT})\b[^>]*>",
    re.IGNORECASE,
)
_BREAK_PRESENTATION_TAG_RE = re.compile(
    r"</?\s*(?:p|div|section|article|br|ol|ul|li|h[1-6]|table|thead|tbody|tfoot|tr|td|th)\b[^>]*>",
    re.IGNORECASE,
)
_NONE_PLACEHOLDER_RE = re.compile(r"^(?:None\s*){2,}$", re.IGNORECASE)
_ESCAPED_LEADING_MARKER_RE = re.compile(r"(?m)^\s*\\[*_`]+\s*")
_TOKEN_ONLY_DEBRIS_RE = re.compile(r"^[\\|<>/_* -]{1,32}$")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

_PERSIAN_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")

_IMAGE_DESCRIPTION_STRONG_RE = re.compile(
    r"""(?ix)
    \b(?:
        (?:the|this|an?|a)\s+image\b
        | this\s+is\s+(?:a\s+)?(?:photograph|photo|picture|caption)\b
        | this\s+is\s+(?:a\s+)?(?:black\s+and\s+white\s+)?(?:line\s+)?drawing\b
        | caption\s+for\s+the\s+image\b
        | a\s+person\s+is\s+seen\b
        | you\s+can\s+see\b
        | the\s+style\s+of\s+(?:the|this)\s+image\b
        | image\s+(?:displays|depicts|shows|contains|appears|is)\b
    )\b
    """
)
_IMAGE_DESCRIPTION_CONTINUATION_RE = re.compile(
    r"""(?ix)^\s*(?:
        in\s+the\s+(?:background|foreground|midground)
        | behind\s+(?:this|the)
        | the\s+(?:building|structure|scene|sky|ground|overall|red|vibrant|bright|
                    flowers?|poppies?|roof|clouds?|design|icon)
        | there\s+(?:is|are)
        | it\s+(?:is|has|appears)
    )\b
    """
)
_VISUAL_DESCRIPTION_WORDS = frozenset(
    """
    image photograph photo picture scene background foreground midground building
    structure roof sky cloud clouds truck trucks flower flowers poppies petals grass
    foliage fuel pump pumps camera color colors red white blue visible depicts displays
    appears icon icons flame flames smoke design designs gesture hand fingers sunlight
    window windows tank facility hills landscape drawing caption
    """.split()
)


def _visual_description_score(value: str) -> int:
    return sum(
        word.lower() in _VISUAL_DESCRIPTION_WORDS
        for word in _LATIN_WORD_RE.findall(value)
    )


def _looks_like_generated_image_description(
    value: str,
    *,
    previous_was_description: bool,
) -> bool:
    """Identify English visual-caption prose generated by the VLM, not OCR text."""

    if not value.strip() or _PERSIAN_SCRIPT_RE.search(value):
        return False

    words = _LATIN_WORD_RE.findall(value)
    if not words:
        return False

    if _IMAGE_DESCRIPTION_STRONG_RE.search(value):
        return True
    if previous_was_description and _IMAGE_DESCRIPTION_CONTINUATION_RE.search(value):
        return True

    # Covers terse visual captions such as the fire-icon and red-poppies examples
    # observed in production while leaving ordinary English headings/body text alone.
    return len(words) >= 8 and _visual_description_score(value) >= 4


def _strip_control_tokens_from_line(value: str) -> tuple[str, int]:
    """Remove model transport tokens from one line and return removal count."""

    control_count = len(_ANGLE_PIPE_TOKEN_RE.findall(value))
    control_count += len(_BARE_PIPE_TOKEN_RE.findall(value))
    control_count += len(_XML_ROLE_TOKEN_RE.findall(value))

    cleaned = _ANGLE_PIPE_TOKEN_RE.sub("", value)
    cleaned = _BARE_PIPE_TOKEN_RE.sub("", cleaned)
    cleaned = _XML_ROLE_TOKEN_RE.sub("", cleaned)
    cleaned = _ROLE_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.strip()

    if _STANDALONE_ROLE_RE.match(cleaned):
        cleaned = ""
    if _TOKEN_ONLY_DEBRIS_RE.fullmatch(cleaned):
        cleaned = ""

    # Lines such as ``0</|im_start|>|user|l3</|ref|>`` are template debris.
    # Preserve logo/label text after a single token, but discard short leftovers
    # when multiple control tokens prove the whole line came from the template.
    if (
        control_count >= 2
        and not _PERSIAN_SCRIPT_RE.search(cleaned)
        and len(re.sub(r"\W", "", cleaned)) <= 8
    ):
        cleaned = ""

    return cleaned, control_count


def strip_vlm_artifacts(value: str) -> str:
    """Remove VLM-only artifacts while preserving recognized lexical text.

    This function deliberately does not perform Persian Unicode normalization; it is
    also used to produce artifact-free ``raw_text``. Exact native Ollama JSON remains
    available in ``WIKI_HAMI_VLM_DIAGNOSTICS_DIR``.
    """

    cleaned = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("｜", "|")
    cleaned = _DATA_URI_RE.sub("\n", cleaned)

    # Turn presentation block tags into line boundaries before stripping the rest.
    cleaned = _BREAK_PRESENTATION_TAG_RE.sub("\n", cleaned)
    cleaned = _PRESENTATION_TAG_RE.sub("", cleaned)

    processed_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()

        if (
            _PDF_TOOL_RE.match(line)
            or _META_MARKER_RE.match(line)
            or _GROUNDING_IMAGE_RE.match(line)
        ):
            processed_lines.append("")
            continue

        line, _ = _strip_control_tokens_from_line(line)
        line = _ESCAPED_LEADING_MARKER_RE.sub("", line).strip()
        if _NONE_PLACEHOLDER_RE.fullmatch(line):
            line = ""
        processed_lines.append(line)

    # Model image-caption prose often spans several HTML paragraphs. Drop only
    # English-only visual-description blocks; Persian/mixed document text is retained.
    blocks = re.split(r"\n\s*\n+", "\n".join(processed_lines))
    kept_blocks: list[str] = []
    previous_was_description = False

    for block in blocks:
        block = block.strip()
        if not block:
            previous_was_description = False
            continue

        if _looks_like_generated_image_description(
            block,
            previous_was_description=previous_was_description,
        ):
            previous_was_description = True
            continue

        previous_was_description = False
        kept_blocks.append(block)

    cleaned = "\n\n".join(kept_blocks)
    cleaned = _EXCESS_BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_vlm_canonical_text(value: str) -> str:
    """Return canonical OCR text with all VLM/template artifacts removed."""

    return normalize_persian_ocr_text(strip_vlm_artifacts(value))


def clean_vlm_lines(lines: list[OCRLine]) -> list[OCRLine]:
    """Clean VLM lines before conversion to persisted OCR objects.

    ``text`` is normalized canonical OCR. ``raw_text`` remains pre-normalization
    lexical OCR, but model transport/image-description artifacts are removed from it.
    Empty artifact-only lines are dropped and explicit reading order is compacted.
    """

    cleaned_lines: list[OCRLine] = []
    for line in lines:
        source_raw = line.raw_text if line.raw_text is not None else line.text
        cleaned_raw = strip_vlm_artifacts(source_raw)
        cleaned_text = clean_vlm_canonical_text(line.text)

        if not cleaned_text:
            # If the normalized ``text`` consisted only of model artifacts, do not
            # persist its corresponding image description/control payload as OCR.
            continue

        cleaned_lines.append(
            replace(
                line,
                text=cleaned_text,
                raw_text=cleaned_raw or cleaned_text,
            )
        )

    if cleaned_lines and all(line.reading_order is not None for line in cleaned_lines):
        cleaned_lines = [
            replace(line, reading_order=index)
            for index, line in enumerate(cleaned_lines)
        ]

    return cleaned_lines
