from scripts.build_grouping_dataset import build_dataset
from tests.contract.test_grouping_training_pair_contract import _record


def test_dataset_detects_duplicates_conflicts_and_splits_by_source():
    rows = [_record("p1", "SAME_GROUP"), _record("p1", "NEW_GROUP"), _record("p2", "NEW_GROUP")]
    rows[2]["source_id"] = "s2"; rows[2]["document_id"] = "d2"
    dataset, summary = build_dataset(rows, validation_ratio=.5, seed=1)
    assert len(dataset) == 2 and any("conflict" in error for error in summary["errors"])
    by_source = {}
    for row in dataset: by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())
