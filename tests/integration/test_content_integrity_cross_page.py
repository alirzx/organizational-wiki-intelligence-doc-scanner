from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from tests.fixtures.grouping.factories import FakeClassifier, make_block


def test_two_page_persian_continuation_and_unrelated_column():
    blocks = [
        make_block("tail", "این بند ادامه دارد", page_number=1, ordinal=5, y1=1300, y2=1340, column_id="left"),
        make_block("head", "و در صفحه بعد تکمیل می‌شود.", page_number=2, ordinal=0, y1=50, y2=90, column_id="left"),
        make_block("other", "متن مستقل", page_number=2, ordinal=1, x1=600, x2=900, y1=50, y2=90, column_id="right"),
    ]
    config = CandidateConfig(reading_lookahead=1, cross_page_window=2, max_page_distance=1)
    candidates = generate_candidates(blocks, config)
    scores = {pair.pair_id: (0.95 if (pair.block_a_id, pair.block_b_id) == ("tail", "head") else 0.1) for pair in candidates}
    result = resolve_paragraphs(blocks, FakeClassifier(scores), candidate_config=config)
    assert [tuple(member.block_id for member in group.members) for group in result.groups] == [("tail", "head"), ("other",)]
    assert result.groups[0].cross_page
