from app.text_processing.linguistic_features import extract_linguistic_features
from app.text_processing.structural_features import extract_structural_features
from tests.fixtures.grouping.factories import make_block


def test_structural_features_cover_markers_heading_and_dimensions():
    a = make_block("a", "۱) نخست", x1=150, y1=100, x2=400, y2=130)
    b = make_block("b", "ادامه بند", ordinal=1, x1=180, y1=140, x2=600, y2=170)
    values = extract_structural_features(a, b)
    assert values["a_list_start"] == 1
    assert values["a_numeric_start"] == 1
    assert values["indent_delta_norm"] > 0
    assert values["line_height_ratio"] == 1


def test_structural_features_detect_persian_and_english_alphabetic_markers():
    for marker in ("الف) متن", "A. text", "• متن", "ماده ۱- متن"):
        a = make_block("a", marker)
        b = make_block("b", "continuation", ordinal=1)
        assert extract_structural_features(a, b)["a_list_start"] == 1


def test_linguistic_features_capture_completion_connectors_and_lengths():
    a = make_block("a", "This sentence continues")
    b = make_block("b", "and it ends.", ordinal=1)
    values = extract_linguistic_features(a, b)
    assert values["a_sentence_complete"] == 0
    assert values["b_sentence_complete"] == 1
    assert values["b_continuation_start"] == 1
    assert values["a_token_count"] == 3
    assert values["length_ratio"] > 1
