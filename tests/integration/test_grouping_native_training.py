import json

from app.orchestration.extractor import ExtractionOrchestrator
from app.core.config import Settings
from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
from app.text_processing.feature_extractor import extract_features, feature_schema
from scripts.train_grouping_model import train
from scripts.evaluate_grouping_model import evaluate_documents, evaluate_scenarios
from tests.fixtures.grouping.factories import make_block, make_candidate, FakeClassifier
from tests.unit.test_grouping_regressions import record_for


def training_records():
    rows = []
    for source,split in [('train-doc','train'),('valid-doc','validation'),('test-doc','test')]:
        for i in range(80):
            a = make_block('a','A completed sentence.' if i%2 else 'unfinished paragraph')
            b = make_block('b','Another paragraph.' if i%2 else 'and continued',ordinal=1,
                           y1=450 if i%2 else 140,y2=480 if i%2 else 170)
            rows.append(record_for(a,b)|{'pair_id':f'{source}-{i}','document_id':source,'source_id':source,
                                        'split':split,'label':'NEW_GROUP' if i%2 else 'SAME_GROUP'})
    return rows


def test_real_training_package_batch_inference_and_model_reuse(tmp_path):
    package = train(training_records(),tmp_path/'native')
    metadata = json.loads((package/'metadata.json').read_text())
    assert metadata['feature_schema']['version'] == 'grouping.features.v2'
    classifier = LightGBMRelationshipClassifier(package,feature_schema(),batch_size=2)
    blocks = [make_block(str(i),'unfinished',ordinal=i,y1=40*i+1,y2=40*i+31) for i in range(4)]
    pairs = [make_candidate(a,b) for a,b in zip(blocks,blocks[1:])]
    features = {p.pair_id:extract_features(blocks[i],blocks[i+1],p) for i,p in enumerate(pairs)}
    predictions = classifier.predict_many({b.block_id:b for b in blocks},pairs,features,semantic_available=False)
    singles = [classifier.predict(blocks[i],blocks[i+1],p,features[p.pair_id],semantic_available=False) for i,p in enumerate(pairs)]
    assert predictions == singles
    orchestrator = ExtractionOrchestrator(Settings(_env_file=None,grouping_enabled=True,grouping_model_path=str(package)))
    first = orchestrator._classifier()
    assert first is orchestrator._classifier()
    first._load()
    assert first._load() is first._booster
    # An explicit runtime setting wins over the package threshold.
    explicit = LightGBMRelationshipClassifier(package,feature_schema(),merge_threshold=.9)
    assert explicit._thresholds() == (.9,.55)
    orchestrator.close()


def test_document_evaluation_runs_resolver_and_ignores_cluster_label_names():
    a,b = make_block('a','unfinished'),make_block('b','and continued',ordinal=1,y1=140,y2=170)
    record = record_for(a,b)
    document = {'document_id':'eval-doc','blocks':[record['block_a'],record['block_b']],
                'expected_groups':[{'member_ids':['a','b'],'group_type':'paragraph'}]}
    result = evaluate_documents([document],lambda:FakeClassifier(default=.9))['documents'][0]
    assert result['bcubed_f1'] == 1 and result['ari'] == 1 and result['fragmentation_rate'] == 0
    rows = [{'expected_group':'original','predicted_group':'different-id','expected_type':'paragraph','predicted_type':'paragraph'}]
    assert evaluate_scenarios(rows)['pass_rate'] == 1
