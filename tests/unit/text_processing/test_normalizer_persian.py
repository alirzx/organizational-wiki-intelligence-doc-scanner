import pytest

from app.text_processing.normalizer import TextNormalizer
from app.text_processing.types import NormalizationProfile


@pytest.mark.parametrize("source,expected", [
    ("ي ك", "ی ک"), ("الف\n  ب", "الف ب"), ("الف\u200cب", "الف\u200cب"),
    ("RTL فارسی + LTR API", "RTL فارسی + LTR API"),
])
def test_persian_normalization_is_idempotent_and_preserves_original(source, expected):
    normalizer = TextNormalizer()
    result = normalizer.normalize(source)
    assert result.original == source
    assert result.normalized == expected
    assert normalizer.normalize(result.normalized).normalized == expected


def test_digit_policies():
    assert TextNormalizer(NormalizationProfile(digit_policy="persian")).normalize("12۳").normalized == "۱۲۳"
    assert TextNormalizer(NormalizationProfile(digit_policy="english")).normalize("۱۲٣").normalized == "123"
