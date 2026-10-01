from app.text_processing.content_graph import resolve_components
from app.text_processing.relationship import prediction_from_score
from tests.fixtures.grouping.factories import make_block, make_candidate


def test_threshold_boundaries_and_singletons():
    blocks = [make_block("a", "a"), make_block("b", "b", ordinal=1), make_block("c", "c", ordinal=2)]
    pairs = [make_candidate(blocks[0], blocks[1]), make_candidate(blocks[1], blocks[2])]
    predictions = [prediction_from_score(pairs[0], 0.75), prediction_from_score(pairs[1], 0.55)]
    result = resolve_components(blocks, predictions)
    assert result.components == (("a", "b"), ("c",))
    assert len(result.accepted) == 1 and len(result.uncertain) == 1


def test_transitive_bridge_and_incompatible_columns_are_guarded():
    blocks = [
        make_block("a", "a", column_id="left"),
        make_block("b", "b", ordinal=1, column_id="left"),
        make_block("c", "c", ordinal=2, column_id="right"),
    ]
    pairs = [make_candidate(blocks[0], blocks[1]), make_candidate(blocks[1], blocks[2])]
    predictions = [prediction_from_score(pair, score) for pair, score in zip(pairs, (0.9, 0.85))]
    result = resolve_components(blocks, predictions, max_component_size=2)
    assert result.components == (("a", "b"), ("c",))
    assert result.rejected[0].decision.value == "guard_rejected"
