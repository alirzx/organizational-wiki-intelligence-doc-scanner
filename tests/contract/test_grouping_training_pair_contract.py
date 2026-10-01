import json
from pathlib import Path

import jsonschema

from scripts.build_grouping_dataset import validate_record


def _record(pair_id="p1", label="SAME_GROUP"):
    block = {"block_id": "a", "text": "text", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
             "page_id": "d:p1", "page_number": 1, "content_ordinal": 0, "page_width": 100, "page_height": 100, "confidence": .9}
    return {"schema_version": "wiki-hami.grouping-pair.v1", "pair_id": pair_id, "document_id": "d", "source_id": "s",
            "block_a": block, "block_b": block | {"block_id": "b", "content_ordinal": 1},
            "candidate_reasons": ["manual"], "label": label, "split": "unassigned"}


def test_training_pair_json_schema_and_semantic_geometry():
    schema = json.loads(Path("specs/001-content-integrity-grouping/contracts/grouping-training-pair.schema.json").read_text(encoding="utf-8"))
    record = _record(); jsonschema.validate(record, schema)
    assert validate_record(record) == []
    invalid = _record(); invalid["block_a"]["bbox"]["x2"] = 0
    assert "invalid:block_a:bbox" in validate_record(invalid)


def test_jsonl_is_human_reviewable():
    encoded = json.dumps(_record(), ensure_ascii=False)
    assert "pair_id" in encoded and "block_a" in encoded and "SAME_GROUP" in encoded
