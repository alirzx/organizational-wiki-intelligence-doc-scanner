import math
from dataclasses import replace
import pytest

from app.text_processing.candidate_generator import CandidateConfig, generate_candidate_batch
from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
from app.text_processing.classifiers.base import GroupingModelIncompatible
from app.text_processing.content_graph import resolve_components
from app.text_processing.feature_extractor import extract_features, feature_schema, precompute_block_features
from app.text_processing.relationship import prediction_from_score
from scripts.train_grouping_model import train
from tests.fixtures.grouping.factories import make_block, make_candidate
from tests.unit.test_grouping_regressions import record_for


def test_cached_features_match_uncached_features():
    blocks = [make_block('a','1. List item'),make_block('b','and continued.',ordinal=1)]
    pair = make_candidate(*blocks)
    cached = extract_features(*blocks,pair,block_features=precompute_block_features(blocks)).values
    plain = extract_features(*blocks,pair).values
    assert all(x==y or math.isnan(x) and math.isnan(y) for x,y in zip(cached,plain))


def test_global_optional_budget_does_not_starve_pages():
    blocks = [make_block(str(i),'unfinished',page_number=i//50+1,ordinal=i%50,y1=24*(i%50)+1,y2=24*(i%50)+20) for i in range(1000)]
    result = generate_candidate_batch(blocks,CandidateConfig(max_pairs=1))
    assert len(result.diagnostics['per_page']) == 20
    for page in range(1,21):
        ids = {(p.block_a_id,p.block_b_id) for p in result.pairs}
        start = (page-1)*50
        assert all((str(i),str(i+1)) in ids for i in range(start,start+49))


def test_explicit_separate_edge_blocks_transitive_bridge():
    blocks = [make_block(str(i),'text',ordinal=i) for i in range(3)]
    pairs = [make_candidate(blocks[0],blocks[1]),make_candidate(blocks[1],blocks[2]),make_candidate(blocks[0],blocks[2])]
    result = resolve_components(blocks,[prediction_from_score(p,s) for p,s in zip(pairs,[.95,.9,.1])],candidates=pairs)
    assert len(result.components) == 2
    assert any('explicit_separation' in p.guard_reasons for p in result.rejected)


def test_custom_candidate_schema_is_resolved_without_hash_reconstruction():
    blocks = [make_block('a','a'),make_block('b','b',ordinal=1)]
    pair = generate_candidate_batch(blocks,CandidateConfig(schema_version='custom.v9')).pairs[0]
    assert resolve_components(blocks,[prediction_from_score(pair,.9)],candidates=[pair]).components == (('a','b'),)


@pytest.mark.parametrize('scores', [[float('nan')],[float('inf')],[],[.5,.6],[[.5,.5]]])
def test_invalid_native_scores_fail(scores):
    class Booster:
        def predict(self,*args,**kwargs): return scores
    c = object.__new__(LightGBMRelationshipClassifier)
    c.schema = feature_schema(); c.manifest = {'package_id':'bad'}; c._booster = Booster()
    a,b = make_block('a','a'),make_block('b','b',ordinal=1)
    pair = make_candidate(a,b)
    with pytest.raises(GroupingModelIncompatible):
        c.predict(a,b,pair,extract_features(a,b,pair),semantic_available=False)


def test_training_rejects_source_leakage_before_loading_library(tmp_path):
    a,b = make_block('a','a'),make_block('b','b',ordinal=1)
    row = record_for(a,b)
    with pytest.raises(ValueError,match='disjoint'):
        train([row,row|{'split':'validation'}],tmp_path/'model')


def test_nonadjacent_merge_cannot_jump_over_explicit_heading():
    blocks = [make_block('a','body'),replace(make_block('title','Title',ordinal=1),block_type='heading'),make_block('b','other body',ordinal=2)]
    pair = make_candidate(blocks[0],blocks[2])
    result = resolve_components(blocks,[prediction_from_score(pair,.99)],candidates=[pair])
    assert len(result.components) == 3
    assert result.rejected[0].guard_reasons == ('heading_boundary',)


def test_interleaved_columns_keep_local_continuity_and_separation():
    from app.orchestration.paragraph_resolver import resolve_paragraphs
    from app.text_processing.classifiers.clustering_classifier import ClusteringRelationshipClassifier
    blocks = [make_block('l1','unfinished left',ordinal=0,column_id='left'),
              make_block('r1','unfinished right',ordinal=1,column_id='right',x1=650,x2=950),
              make_block('l2','and left continues',ordinal=2,column_id='left',y1=140,y2=170),
              make_block('r2','and right continues',ordinal=3,column_id='right',x1=650,x2=950,y1=140,y2=170)]
    result = resolve_paragraphs(blocks,ClusteringRelationshipClassifier())
    assert [{m.block_id for m in g.members} for g in result.groups] == [{'l1','l2'},{'r1','r2'}]


def test_zero_optional_budget_preserves_cross_page_candidate():
    a = make_block('tail','unfinished',y1=1300,y2=1340)
    b = make_block('head','and continued',page_number=2,y1=40,y2=70)
    result = generate_candidate_batch([a,b],CandidateConfig(max_pairs=0))
    assert [(p.block_a_id,p.block_b_id) for p in result.pairs] == [('tail','head')]
    assert result.diagnostics['boundary_candidate_recall'] == 1


def test_named_merge_score_is_independent_of_probability_dictionary_order():
    from app.text_processing.types import RelationshipLabel
    a,b,c = [make_block(str(i),'text',ordinal=i) for i in range(3)]
    pairs = [make_candidate(a,b),make_candidate(b,c)]
    high,low = prediction_from_score(pairs[0],.95),prediction_from_score(pairs[1],.85)
    high = replace(high,probabilities={RelationshipLabel.NEW_GROUP:.05,RelationshipLabel.SAME_GROUP:.95})
    result = resolve_components([a,b,c],[low,high],candidates=pairs,max_component_size=2)
    assert result.components == (('0','1'),('2',))
