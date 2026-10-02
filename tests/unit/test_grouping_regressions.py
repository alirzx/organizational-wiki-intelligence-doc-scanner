"""Regression requirements from PERFORMORT.md; synthetic labels are not accuracy evidence."""
import math

import pytest

from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
from app.text_processing.content_graph import resolve_components
from app.text_processing.feature_extractor import extract_features, feature_schema
from scripts.evaluate_grouping_model import evaluate_scenarios
from scripts.train_grouping_model import _row_features
from tests.fixtures.grouping.factories import make_block, make_candidate


def record_for(a, b):
    def serialize(block):
        return {**block.__dict__, 'bbox': block.bbox.model_dump(), 'text': block.original_text,
                'confidence': block.ocr_confidence}
    return {'pair_id': make_candidate(a, b).pair_id, 'document_id': a.document_id,
            'source_id': a.document_id, 'block_a': serialize(a), 'block_b': serialize(b),
            'candidate_reasons': ['reading_window'], 'label': 'SAME_GROUP', 'split': 'train'}


@pytest.mark.parametrize('page,semantic', [(1, None), (2, None), (1, .8)])
def test_training_features_equal_runtime_including_missing_values(page, semantic):
    a = make_block('a', '1. First item')
    b = make_block('b', 'and more details', ordinal=1, page_number=page, y1=140, y2=170)
    record = record_for(a, b)
    record['semantic_similarity'] = semantic
    trained = _row_features(record)
    runtime = extract_features(a, b, make_candidate(a, b), semantic_similarity=semantic).values
    assert len(trained) == len(runtime)
    assert all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(trained, runtime))


def test_candidate_cap_preserves_all_eligible_adjacency_on_dense_page():
    blocks = [make_block(str(i), 'unfinished text', ordinal=i, y1=24*i+1, y2=24*i+20) for i in range(500)]
    pairs = generate_candidates(blocks, CandidateConfig(max_pairs=17))
    endpoints = {(p.block_a_id, p.block_b_id) for p in pairs}
    assert all((str(i), str(i+1)) in endpoints for i in range(499))


def test_zero_edge_graph_does_not_hash_any_pairs(monkeypatch):
    def forbidden(*args):
        raise AssertionError('graph must not reconstruct pair hashes')
    monkeypatch.setattr('app.text_processing.ids.stable_pair_id', forbidden)
    blocks = [make_block(str(i), 'text', ordinal=i) for i in range(1000)]
    result = resolve_components(blocks, [])
    assert len(result.components) == len(blocks)


def test_package_thresholds_apply_to_lightgbm_prediction():
    class Booster:
        def predict(self, rows, **kwargs):
            return [.6] * len(rows)
    classifier = object.__new__(LightGBMRelationshipClassifier)
    classifier.schema = feature_schema()
    classifier.manifest = {'package_id': 'probe', 'thresholds': {'merge': .5, 'uncertain_lower': .3}}
    classifier._booster = Booster()
    a, b = make_block('a', 'unfinished'), make_block('b', 'continued', ordinal=1)
    pair = make_candidate(a, b)
    assert classifier.predict(a, b, pair, extract_features(a,b,pair), semantic_available=False).decision.value == 'merge'


def test_evaluator_rejects_missing_predictions():
    with pytest.raises(ValueError):
        evaluate_scenarios([{'scenario': 'missing'}])


def test_evaluator_wrong_group_type_cannot_pass():
    report = evaluate_scenarios([{'scenario': 'wrong-type', 'predicted_group': 'g', 'expected_group': 'g',
                                 'predicted_type': 'list', 'expected_type': 'paragraph'}])
    assert report['pass_rate'] == 0
