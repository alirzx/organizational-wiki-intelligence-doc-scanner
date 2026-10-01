from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from app.text_processing.types import NormalizationProfile


_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+")
_LINEBREAKS = re.compile(r"\r\n?|\n+")
_ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_ENGLISH_DIGITS = "0123456789"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    profile_version: str


class TextNormalizer:
    def __init__(self, profile: NormalizationProfile | None = None):
        self.profile = profile or NormalizationProfile()

    def normalize(self, text: str) -> NormalizedText:
        value = unicodedata.normalize(self.profile.unicode_form, text)
        value = value.translate(_ARABIC_TO_PERSIAN)
        if self.profile.normalize_half_space:
            value = re.sub(r"[\u200b\u200d\u2060]", "", value)
            value = re.sub(r"\s*\u200c\s*", "\u200c", value)
        if self.profile.digit_policy == "persian":
            value = value.translate(str.maketrans(_ENGLISH_DIGITS + _ARABIC_DIGITS, _PERSIAN_DIGITS * 2))
        elif self.profile.digit_policy == "english":
            value = value.translate(str.maketrans(_PERSIAN_DIGITS + _ARABIC_DIGITS, _ENGLISH_DIGITS * 2))
        if self.profile.collapse_whitespace:
            value = _LINEBREAKS.sub(" ", value)
            value = _HORIZONTAL_WHITESPACE.sub(" ", value)
        value = value.strip()
        return NormalizedText(
            original=text,
            normalized=value,
            profile_version=self.profile.version,
        )
