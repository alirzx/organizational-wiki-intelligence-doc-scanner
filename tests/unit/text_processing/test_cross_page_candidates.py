from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from app.text_processing.types import CandidateReason
from tests.fixtures.grouping.factories import make_block


def test_cross_page_candidates_are_bounded_and_not_automatic_merges():
    blocks = [
        make_block("tail", "continues", page_number=1, ordinal=9, y1=1300, y2=1340, column_id="left"),
        make_block("head", "و ادامه", page_number=2, ordinal=0, y1=50, y2=90, column_id="left"),
        make_block("other", "other", page_number=2, ordinal=1, y1=50, y2=90, x1=600, x2=900, column_id="right"),
        make_block("far", "far", page_number=3, ordinal=0, column_id="left"),
    ]
    pairs = generate_candidates(blocks, CandidateConfig(cross_page_window=1, max_page_distance=1))
    boundary = [pair for pair in pairs if pair.cross_page]
    assert [(pair.block_a_id, pair.block_b_id) for pair in boundary] == [("tail", "head")]
    assert CandidateReason.PAGE_BOUNDARY in boundary[0].reasons
    assert all(pair.page_distance == 1 for pair in boundary)
