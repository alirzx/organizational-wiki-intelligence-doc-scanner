from scripts.evaluate_grouping_model import evaluate_scenarios
from scripts.export_grouping_review import export_review


def test_review_export_is_private_and_scenario_evaluation_is_explainable():
    review = export_review([{"decision": "uncertain", "pair_id": "p", "embedding": [1, 2], "context": "text"}])
    assert review == [{"decision": "uncertain", "pair_id": "p", "context": "text"}]
    report = evaluate_scenarios([{"scenario": "continuation", "predicted_group": "g", "expected_group": "g",
                                  "predicted_type": "paragraph", "expected_type": "paragraph"}])
    assert report["pass_rate"] == 1 and report["scenarios"]["continuation"]["passed"]
