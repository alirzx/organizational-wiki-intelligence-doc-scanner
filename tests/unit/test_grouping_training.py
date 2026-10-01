from scripts.train_grouping_model import compute_metrics


def test_training_metrics_are_complete_and_seed_independent():
    metrics, matrix = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [.1, .7, .8, .9], candidate_recall=.95)
    for key in ("new_group_precision", "new_group_recall", "same_group_precision", "same_group_recall", "macro_f1", "weighted_f1", "pr_auc", "roc_auc", "candidate_recall"):
        assert key in metrics
    assert matrix["labels"] == ["NEW_GROUP", "SAME_GROUP"]
