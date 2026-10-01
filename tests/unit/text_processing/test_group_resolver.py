from app.text_processing.group_resolver import resolve_groups
from tests.fixtures.grouping.factories import make_block


def test_group_resolution_assigns_types_roles_text_ids_and_bbox():
    blocks = [
        make_block("h", "عنوان", x1=100, y1=100, x2=300, y2=140),
        make_block("i", "۱) مورد", ordinal=1, x1=120, y1=150, x2=400, y2=180),
        make_block("b", "توضیح", ordinal=2, x1=140, y1=190, x2=500, y2=220),
    ]
    groups = resolve_groups(blocks, (("h", "i", "b"),), edge_confidences={"h:i": 0.9, "i:b": 0.8})
    group = groups[0]
    assert group.group_order == 0
    assert group.group_type.value in {"list", "list_section"}
    assert [member.member_order for member in group.members] == [0, 1, 2]
    assert group.page_spans[0].bbox.x1 == 100
    assert group.page_spans[0].bbox.x2 == 500
    assert group.confidence_method
    assert resolve_groups(blocks, (("h", "i", "b"),))[0].group_id == group.group_id


def test_singletons_are_retained():
    block = make_block("a", "standalone")
    assert resolve_groups([block], (("a",),))[0].text == "standalone"
