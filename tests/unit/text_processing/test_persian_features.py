from app.text_processing.linguistic_features import extract_linguistic_features
from app.text_processing.structural_features import is_list_start
from tests.fixtures.grouping.factories import make_block


def test_persian_connectors_and_sentence_continuation():
    values = extract_linguistic_features(make_block("a", "متن ادامه دارد"), make_block("b", "و تکمیل می‌شود.", ordinal=1))
    assert values["b_continuation_start"] == 1
    assert values["a_sentence_complete"] == 0


def test_persian_markers_do_not_misclassify_ordinary_vav():
    assert is_list_start("الف) نخست")
    assert is_list_start("۲- دوم")
    assert is_list_start("• سوم")
    assert not is_list_start("و این یک جمله عادی است")
