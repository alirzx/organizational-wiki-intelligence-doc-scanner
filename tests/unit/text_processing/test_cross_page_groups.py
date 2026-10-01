from app.modules.ocr.adapter import groups_to_detected_objects
from app.text_processing.group_resolver import resolve_groups
from tests.fixtures.grouping.factories import make_block


def test_cross_page_group_has_local_spans_and_shared_projection_id():
    blocks = [
        make_block("tail", "first", page_number=1, ordinal=9, x1=100, y1=1300, x2=500, y2=1340),
        make_block("head", "second", page_number=2, ordinal=0, x1=120, y1=50, x2=520, y2=90),
    ]
    group = resolve_groups(blocks, (("tail", "head"),))[0]
    assert group.cross_page and [span.page_number for span in group.page_spans] == [1, 2]
    assert group.page_spans[0].bbox.y1 == 1300 and group.page_spans[1].bbox.y1 == 50
    objects = groups_to_detected_objects([group], document_id="doc-test", backend_name="test", model_id="test")
    assert len(objects) == 2
    assert {obj.metadata["group_id"] for obj in objects} == {group.group_id}
    assert [obj.metadata["group_span_index"] for obj in objects] == [0, 1]
