from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from app.text_processing.types import CandidateReason
from tests.fixtures.grouping.factories import make_block


def test_candidates_are_deterministic_deduplicated_and_reason_merged():
    blocks = [
        make_block("a", "1. item", ordinal=0, y1=100, y2=130),
        make_block("b", "continuation", ordinal=1, y1=140, y2=170),
        make_block("c", "next", ordinal=2, y1=180, y2=210),
    ]
    first = generate_candidates(blocks, CandidateConfig(reading_lookahead=2, max_pairs=10))
    second = generate_candidates(list(reversed(blocks)), CandidateConfig(reading_lookahead=2, max_pairs=10))
    assert first == second
    assert len({pair.pair_id for pair in first}) == len(first)
    assert CandidateReason.READING_WINDOW in first[0].reasons
    assert tuple(first[0].reasons) == tuple(sorted(first[0].reasons, key=str))


def test_candidate_growth_is_bounded():
    blocks = [make_block(str(i), f"line {i}", ordinal=i, y1=10*i+1, y2=10*i+9) for i in range(100)]
    assert len(generate_candidates(blocks, CandidateConfig(max_pairs=17))) == 17
