import math

from app.text_processing.visual_features import extract_visual_features
from tests.fixtures.grouping.factories import make_block


def test_visual_features_are_page_relative_and_geometric():
    left = make_block("a", "one", x1=100, y1=100, x2=500, y2=150)
    right = make_block("b", "two", ordinal=1, x1=120, y1=170, x2=520, y2=220)
    values = extract_visual_features(left, right)
    assert values["vertical_gap_norm"] == 20 / 1400
    assert values["horizontal_gap_norm"] == 0
    assert values["horizontal_overlap_ratio"] == 380 / 400
    assert values["left_alignment_delta_norm"] == 20 / 1000
    assert values["indentation_delta_norm"] == 20 / 1000
    assert values["width_ratio"] == 1
    assert values["height_ratio"] == 1
    assert values["same_column"] == 1
    assert values["same_page"] == 1
    assert values["page_distance"] == 0
    assert all(math.isfinite(value) for value in values.values())


def test_visual_features_represent_missing_column_natively():
    a = make_block("a", "a", column_id=None)
    b = make_block("b", "b", ordinal=1, column_id=None)
    assert math.isnan(extract_visual_features(a, b)["same_column"])
