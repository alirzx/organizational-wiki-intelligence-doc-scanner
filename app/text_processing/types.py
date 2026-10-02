from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from hashlib import sha256
import json
import math
from typing import Any

from app.schemas.common import BBox, Polygon


class CandidateReason(StrEnum):
    READING_WINDOW = "reading_window"
    SAME_COLUMN = "same_column"
    SPATIAL_NEAR = "spatial_near"
    STRUCTURAL_MATCH = "structural_match"
    PAGE_BOUNDARY = "page_boundary"
    MANUAL = "manual"
    ADJACENT_SAME_PAGE = "adjacent_same_page"
    ADJACENT_CROSS_PAGE = "adjacent_cross_page"
    SAME_COLUMN_NEARBY = "same_column_nearby"
    HEADING_TO_BODY = "heading_to_body"
    LIST_CONTINUATION = "list_continuation"
    GEOMETRY_OVERLAP = "geometry_overlap"


class RelationshipLabel(StrEnum):
    SAME_GROUP = "SAME_GROUP"
    NEW_GROUP = "NEW_GROUP"


class RelationshipDecision(StrEnum):
    MERGE = "merge"
    UNCERTAIN = "uncertain"
    SEPARATE = "separate"
    GUARD_REJECTED = "guard_rejected"


class GroupType(StrEnum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_SECTION = "list_section"
    HEADING_SECTION = "heading_section"
    GENERIC_GROUP = "generic_group"


class MemberRole(StrEnum):
    HEADING = "heading"
    LIST_ITEM = "list_item"
    BODY = "body"
    FIELD = "field"
    CONTINUATION = "continuation"
    UNKNOWN = "unknown"


class GroupingMode(StrEnum):
    CLUSTERED = "clustered"
    LEARNED = "learned"
    LEARNED_WITHOUT_SEMANTICS = "learned_without_semantics"
    HEURISTIC_DISABLED = "heuristic_disabled"
    HEURISTIC_FALLBACK = "heuristic_fallback"


@dataclass(frozen=True)
class NormalizationProfile:
    version: str = "normalizer.v1"
    unicode_form: str = "NFKC"
    digit_policy: str = "preserve"
    mixed_direction_policy: str = "preserve_marks.v1"
    collapse_whitespace: bool = True
    normalize_half_space: bool = True

    def __post_init__(self) -> None:
        if self.unicode_form not in {"NFC", "NFKC"}:
            raise ValueError("unicode_form must be NFC or NFKC")
        if self.digit_policy not in {"preserve", "persian", "english"}:
            raise ValueError("digit_policy must be preserve, persian, or english")


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    document_id: str
    page_id: str
    page_number: int
    content_ordinal: int
    original_text: str
    normalized_text: str
    bbox: BBox
    page_width: int
    page_height: int
    polygon: Polygon | None = None
    ocr_confidence: float | None = None
    block_type: str | None = None
    column_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.block_id or not self.document_id or not self.page_id:
            raise ValueError("block, document, and page IDs must not be empty")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if self.content_ordinal < 0:
            raise ValueError("content_ordinal must be >= 0")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("page dimensions must be positive")
        if self.bbox.width <= 0 or self.bbox.height <= 0:
            raise ValueError("bbox must have positive extent")
        if self.ocr_confidence is not None and not 0 <= self.ocr_confidence <= 1:
            raise ValueError("ocr_confidence must be within [0, 1]")
        if any("embedding" in str(key).lower() for key in self.metadata):
            raise ValueError("block metadata must not contain embeddings")

    @property
    def document_order(self) -> tuple[int, int, str]:
        return (self.page_number, self.content_ordinal, self.block_id)


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    block_a_id: str
    block_b_id: str
    cross_page: bool
    page_distance: int
    reasons: tuple[CandidateReason, ...]

    def __post_init__(self) -> None:
        if not self.pair_id or self.block_a_id == self.block_b_id:
            raise ValueError("candidate pair requires a stable ID and distinct blocks")
        if self.page_distance < 0:
            raise ValueError("page_distance must be >= 0")
        normalized = tuple(sorted(set(self.reasons), key=str))
        if not normalized:
            raise ValueError("candidate reasons must not be empty")
        object.__setattr__(self, "reasons", normalized)


@dataclass(frozen=True)
class FeatureDescriptor:
    name: str
    dtype: str = "float64"
    missing_policy: str = "forbidden"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name must not be empty")
        if self.missing_policy not in {
            "forbidden",
            "native_missing",
            "zero",
            "explicit_flag",
        }:
            raise ValueError("unsupported missing policy")


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    normalizer_version: str
    features: tuple[FeatureDescriptor, ...]
    sha256: str = ""

    def __post_init__(self) -> None:
        names = [item.name for item in self.features]
        if not self.version or not self.features or len(names) != len(set(names)):
            raise ValueError("feature schema requires a version and unique features")
        payload = json.dumps(
            {
                "version": self.version,
                "normalizer_version": self.normalizer_version,
                "features": [item.__dict__ for item in self.features],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = sha256(payload).hexdigest()
        if self.sha256 and self.sha256 != digest:
            raise ValueError("feature schema hash does not match its contents")
        object.__setattr__(self, "sha256", digest)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.features)

    @cached_property
    def semantic_indices(self):
        if 'semantic_similarity' not in self.names or 'semantic_available' not in self.names:
            return None
        return self.names.index('semantic_similarity'), self.names.index('semantic_available')


@dataclass(frozen=True)
class FeatureVector:
    pair_id: str
    feature_schema_hash: str
    values: tuple[float, ...]

    def validate(self, schema: FeatureSchema) -> None:
        if self.feature_schema_hash != schema.sha256:
            raise ValueError("feature schema hash mismatch")
        if len(self.values) != len(schema.features):
            raise ValueError("feature vector length mismatch")
        for value, descriptor in zip(self.values, schema.features, strict=True):
            if math.isnan(value) and descriptor.missing_policy != "native_missing":
                raise ValueError(f"feature {descriptor.name} does not permit missing values")
            if not math.isnan(value) and not math.isfinite(value):
                raise ValueError(f"feature {descriptor.name} must be finite")
        if schema.semantic_indices is not None:
            similarity = self.values[schema.semantic_indices[0]]
            available = self.values[schema.semantic_indices[1]]
            if math.isnan(similarity) != (available == 0):
                raise ValueError("missing semantic similarity requires semantic_available=0")


@dataclass(frozen=True)
class RelationshipPrediction:
    pair_id: str
    label: RelationshipLabel
    confidence: float
    probabilities: dict[RelationshipLabel, float]
    decision: RelationshipDecision
    semantic_available: bool
    model_package_id: str
    guard_reasons: tuple[str, ...] = ()
    block_a_id: str | None = None
    block_b_id: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("prediction confidence must be within [0, 1]")
        if set(self.probabilities) != set(RelationshipLabel):
            raise ValueError("probabilities must contain every relationship label")
        if any(not 0 <= value <= 1 for value in self.probabilities.values()):
            raise ValueError("probabilities must be within [0, 1]")
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")


@dataclass(frozen=True)
class ContentGroupMember:
    block_id: str
    member_order: int
    page_id: str
    page_number: int
    role: MemberRole
    original_text: str
    normalized_text: str
    bbox: BBox
    confidence: float | None

    def __post_init__(self) -> None:
        if self.member_order < 0 or self.page_number < 1:
            raise ValueError("member order and page number are invalid")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("member confidence must be within [0, 1]")


@dataclass(frozen=True)
class PageSpan:
    page_id: str
    page_number: int
    bbox: BBox
    member_ids: tuple[str, ...]
    text: str
    raw_text: str

    def __post_init__(self) -> None:
        if self.page_number < 1 or not self.member_ids:
            raise ValueError("page span requires a valid page and non-empty members")
        if self.bbox.width <= 0 or self.bbox.height <= 0:
            raise ValueError("page span bbox must have positive extent")


@dataclass(frozen=True)
class ContentGroup:
    group_id: str
    group_order: int
    group_type: GroupType
    members: tuple[ContentGroupMember, ...]
    page_spans: tuple[PageSpan, ...]
    text: str
    raw_text: str
    confidence: float
    confidence_method: str
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.group_id or self.group_order < 0 or not self.members or not self.page_spans:
            raise ValueError("content group requires ID, non-negative order, members, and spans")
        if not 0 <= self.confidence <= 1:
            raise ValueError("group confidence must be within [0, 1]")
        if tuple(item.member_order for item in self.members) != tuple(range(len(self.members))):
            raise ValueError("member order must be contiguous from zero")
        all_ids = [item.block_id for item in self.members]
        span_ids = [block_id for span in self.page_spans for block_id in span.member_ids]
        if all_ids != span_ids:
            raise ValueError("page spans must partition members in member order")
        pages = [span.page_number for span in self.page_spans]
        if pages != sorted(pages) or len(pages) != len(set(pages)):
            raise ValueError("page spans must be unique and ordered")
        member_pages = {member.block_id: member.page_number for member in self.members}
        if any(member_pages[item] != span.page_number for span in self.page_spans for item in span.member_ids):
            raise ValueError("page spans may contain only members from their page")

    @property
    def cross_page(self) -> bool:
        return len(self.page_spans) > 1


@dataclass
class GroupingDiagnostics:
    mode: GroupingMode = GroupingMode.HEURISTIC_DISABLED
    fallback_reason: str | None = None
    model_package_id: str | None = None
    feature_schema_version: str | None = None
    semantic_mode: str = "disabled"
    counts: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
