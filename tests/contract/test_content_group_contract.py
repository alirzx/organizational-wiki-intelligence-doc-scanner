from app.schemas.detection import ObjectType
from app.schemas.extraction import DocumentExtractionResponse, GroupingDiagnostics


def test_content_group_contract_is_additive_and_object_types_unchanged():
    assert {item.value for item in ObjectType} == {"paragraph", "table", "figure", "stamp", "signature"}
    assert "content_groups" in DocumentExtractionResponse.model_fields
    assert "grouping" in DocumentExtractionResponse.model_fields
    assert GroupingDiagnostics().model_dump(mode="json")["mode"] == "heuristic_disabled"


def test_grouping_diagnostics_serialization_is_deterministic():
    diagnostic = GroupingDiagnostics(counts={"blocks": 2, "groups": 1})
    assert diagnostic.model_dump_json() == diagnostic.model_dump_json()
