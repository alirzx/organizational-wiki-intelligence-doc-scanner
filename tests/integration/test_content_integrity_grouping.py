from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from tests.fixtures.grouping.factories import FakeClassifier, make_block


def test_learned_path_preserves_logical_units_with_injected_classifier():
    blocks = [
        make_block("heading", "Section", ordinal=0, y1=100, y2=130),
        make_block("item", "1. First item", ordinal=1, y1=140, y2=170),
        make_block("continuation", "continued text", ordinal=2, y1=180, y2=210),
        make_block("separate", "Independent paragraph.", ordinal=3, y1=280, y2=310),
    ]
    candidates = generate_candidates(blocks, CandidateConfig(reading_lookahead=1))
    scores = {candidate.pair_id: (0.9 if candidate.block_b_id != "separate" else 0.1) for candidate in candidates}
    result = resolve_paragraphs(blocks, FakeClassifier(scores), candidate_config=CandidateConfig(reading_lookahead=1))
    assert [len(group.members) for group in result.groups] == [3, 1]
    assert result.groups[0].group_type.value in {"list", "list_section"}
    assert result.groups[1].text == "Independent paragraph."
