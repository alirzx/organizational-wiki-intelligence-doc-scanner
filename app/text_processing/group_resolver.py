from __future__ import annotations

from collections import defaultdict

from app.schemas.common import BBox
from app.text_processing.ids import stable_group_id
from app.text_processing.structural_features import is_heading_like, is_list_start
from app.text_processing.types import ContentGroup, ContentGroupMember, GroupType, MemberRole, PageSpan, TextBlock


def _role(block: TextBlock, index: int, size: int) -> MemberRole:
    if is_list_start(block.normalized_text):
        return MemberRole.LIST_ITEM
    if index == 0 and size > 1 and is_heading_like(block.normalized_text):
        return MemberRole.HEADING
    return MemberRole.BODY if index == 0 else MemberRole.CONTINUATION


def resolve_groups(
    blocks: list[TextBlock],
    components: tuple[tuple[str, ...], ...],
    *,
    edge_confidences: dict[tuple[str, str] | str, float] | None = None,
) -> list[ContentGroup]:
    by_id = {block.block_id: block for block in blocks}
    component_ids = [item for component in components for item in component]
    if len(component_ids) != len(set(component_ids)) or set(component_ids) != set(by_id):
        raise ValueError("components must partition every block exactly once")
    edge_confidences = edge_confidences or {}
    groups: list[ContentGroup] = []
    ordered_components = sorted(components, key=lambda ids: min(by_id[item].document_order for item in ids))
    component_for = {item:index for index,ids in enumerate(ordered_components) for item in ids}
    confidences = defaultdict(list)
    for edge,value in edge_confidences.items():
        if isinstance(edge,tuple) and edge[0] in component_for and edge[1] in component_for:
            index = component_for[edge[0]]
            if index == component_for[edge[1]]:
                confidences[index].append(value)
    for group_order, ids in enumerate(ordered_components):
        selected = sorted((by_id[item] for item in ids), key=lambda item: item.document_order)
        members = tuple(
            ContentGroupMember(
                block_id=block.block_id, member_order=index, page_id=block.page_id, page_number=block.page_number,
                role=_role(block, index, len(selected)), original_text=block.original_text,
                normalized_text=block.normalized_text, bbox=block.bbox, confidence=block.ocr_confidence,
            )
            for index, block in enumerate(selected)
        )
        page_blocks: dict[int, list[TextBlock]] = defaultdict(list)
        for block in selected:
            page_blocks[block.page_number].append(block)
        spans = tuple(
            PageSpan(
                page_id=items[0].page_id, page_number=page, member_ids=tuple(item.block_id for item in items),
                bbox=BBox(x1=min(i.bbox.x1 for i in items), y1=min(i.bbox.y1 for i in items),
                          x2=max(i.bbox.x2 for i in items), y2=max(i.bbox.y2 for i in items)),
                text="\n".join(i.normalized_text for i in items), raw_text="\n".join(i.original_text for i in items),
            )
            for page, items in sorted(page_blocks.items())
        )
        roles = {member.role for member in members}
        group_type = GroupType.LIST_SECTION if MemberRole.LIST_ITEM in roles and MemberRole.HEADING in roles else (
            GroupType.LIST if MemberRole.LIST_ITEM in roles else GroupType.PARAGRAPH
        )
        applicable = confidences[group_order]
        if not applicable and len(ordered_components) == 1:
            applicable = list(edge_confidences.values())
        confidence = min(applicable) if applicable else (1.0 if len(selected) == 1 else 0.5)
        groups.append(ContentGroup(
            group_id=stable_group_id((item.block_id for item in selected), "group.v1"), group_order=group_order,
            group_type=group_type, members=members, page_spans=spans,
            text="\n".join(item.normalized_text for item in selected), raw_text="\n".join(item.original_text for item in selected),
            confidence=confidence, confidence_method="minimum_accepted_edge" if applicable else "structural_default",
            metadata={"cross_page": len(spans) > 1},
        ))
    return groups
