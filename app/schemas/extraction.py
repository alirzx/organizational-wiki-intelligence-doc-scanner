from typing import Any
from pydantic import BaseModel, Field, model_validator

from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType
from app.schemas.image import ImageMetadata, TransformMetadata
from app.schemas.status import ModuleName, ModuleStatus, ProcessingStatus
from app.text_processing.types import GroupType, GroupingMode, MemberRole


class ContentGroupMember(BaseModel):
    block_id: str = Field(min_length=1)
    member_order: int = Field(ge=0)
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    role: MemberRole
    original_text: str
    normalized_text: str
    bbox: BBox
    confidence: float | None = Field(default=None, ge=0, le=1)


class ContentGroupPageSpan(BaseModel):
    page_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: BBox
    member_ids: list[str] = Field(min_length=1)
    text: str
    raw_text: str

    @model_validator(mode="after")
    def validate_bbox(self) -> "ContentGroupPageSpan":
        if self.bbox.width <= 0 or self.bbox.height <= 0:
            raise ValueError("page span bbox must have positive extent")
        return self


class ContentGroup(BaseModel):
    group_id: str = Field(min_length=1)
    group_order: int = Field(ge=0)
    group_type: GroupType
    text: str
    raw_text: str
    confidence: float = Field(ge=0, le=1)
    confidence_method: str = Field(min_length=1)
    members: list[ContentGroupMember] = Field(min_length=1)
    page_spans: list[ContentGroupPageSpan] = Field(min_length=1)
    cross_page: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_members_and_spans(self) -> "ContentGroup":
        if [member.member_order for member in self.members] != list(range(len(self.members))):
            raise ValueError("member order must be contiguous from zero")
        member_ids = [member.block_id for member in self.members]
        span_ids = [member_id for span in self.page_spans for member_id in span.member_ids]
        if member_ids != span_ids:
            raise ValueError("page spans must partition members in member order")
        member_pages = {member.block_id: member.page_number for member in self.members}
        if any(member_pages[item] != span.page_number for span in self.page_spans for item in span.member_ids):
            raise ValueError("page spans may contain only members from their page")
        page_numbers = [span.page_number for span in self.page_spans]
        if page_numbers != sorted(page_numbers) or len(page_numbers) != len(set(page_numbers)):
            raise ValueError("page spans must be unique and ordered by page number")
        expected_cross_page = len(self.page_spans) > 1
        if self.cross_page != expected_cross_page:
            raise ValueError("cross_page must be true iff multiple page spans exist")
        return self


class GroupingDiagnostics(BaseModel):
    mode: GroupingMode = GroupingMode.HEURISTIC_DISABLED
    fallback_reason: str | None = None
    model_package_id: str | None = None
    feature_schema_version: str | None = None
    semantic_mode: str = "disabled"
    counts: dict[str, int] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)


class ModulePageResponse(BaseModel):
    schema_version: str
    request_id: str
    document_id: str
    page_id: str
    page_number: int
    module: ModuleName
    image: ImageMetadata
    transform: TransformMetadata
    objects: list[DetectedObject] = Field(default_factory=list)
    status: ModuleStatus


class PageExtractionResponse(BaseModel):
    schema_version: str
    request_id: str
    document_id: str
    page_id: str
    page_number: int
    page_metadata: dict[str, Any] = Field(default_factory=dict)
    image: ImageMetadata
    transform: TransformMetadata
    objects: list[DetectedObject] = Field(default_factory=list)
    modules: dict[ModuleName, ModuleStatus]
    processing: ProcessingStatus


class DocumentExtractionResponse(BaseModel):
    schema_version: str
    request_id: str
    document_id: str
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    page_count: int
    pages: list[PageExtractionResponse]
    # Flattened document-level view. Every object still carries page provenance.
    objects: list[DetectedObject]
    object_counts: dict[ObjectType, int]
    processing: ProcessingStatus
    content_groups: list[ContentGroup] = Field(default_factory=list)
    grouping: GroupingDiagnostics = Field(default_factory=GroupingDiagnostics)
