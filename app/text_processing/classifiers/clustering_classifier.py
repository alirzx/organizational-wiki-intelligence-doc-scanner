from __future__ import annotations

import math

from app.text_processing.classifiers.base import BlockRelationshipClassifier, GroupingModelUnavailable
from app.text_processing.feature_extractor import feature_schema
from app.text_processing.relationship import prediction_from_score
from app.text_processing.types import CandidatePair, CandidateReason, FeatureVector, GroupingMode, TextBlock


class ClusteringRelationshipClassifier(BlockRelationshipClassifier):
    """Document-local DBSCAN over sparse, structurally eligible OCR neighbors."""

    grouping_mode = GroupingMode.CLUSTERED

    def __init__(self, *, eps: float = 0.45, min_samples: int = 2,
                 merge_threshold: float = 0.75, uncertain_lower: float = 0.55):
        if not 0 < eps <= 1 or min_samples < 2:
            raise ValueError("clustering requires 0 < eps <= 1 and min_samples >= 2")
        self.eps = eps
        self.min_samples = min_samples
        self.merge_threshold = merge_threshold
        self.uncertain_lower = uncertain_lower
        self.schema = feature_schema()
        self._labels: dict[str, int] = {}
        self._distances: dict[str, float] = {}

    @property
    def package_id(self) -> str:
        return f"dbscan.v1:eps={self.eps:g}:min_samples={self.min_samples}"

    def _distance(self, a, b, candidate, vector, page_bounds):
        vector.validate(self.schema)
        f = dict(zip(self.schema.names, vector.values, strict=True))
        if a.document_id != b.document_id or (
            a.column_id is not None and b.column_id is not None and a.column_id != b.column_id
        ):
            return math.inf
        if f["horizontal_overlap_ratio"] < 0.15:
            return math.inf
        same_list = bool(f["a_list_start"] and f["b_list_start"])
        height = max(1.0, min(a.bbox.height, b.bbox.height))
        if candidate.cross_page:
            if b.page_number != a.page_number + 1 or (
                a.block_id != page_bounds[a.page_number][1]
                or b.block_id != page_bounds[b.page_number][0]
                or a.bbox.y2 < 0.8 * a.page_height or b.bbox.y1 > 0.2 * b.page_height
                or (f["a_sentence_complete"] and not same_list)
                or (f["b_list_start"] and not same_list)
            ):
                return math.inf
            gap = 0.2
        else:
            if b.content_ordinal != a.content_ordinal + 1 and CandidateReason.SAME_COLUMN_NEARBY not in candidate.reasons:
                return math.inf
            gap = max(0.0, b.bbox.y1 - a.bbox.y2) / height
            if b.block_type == "heading" or (
                f["b_heading_like"] and not f["b_list_start"]
                and b.bbox.height > 1.3 * a.bbox.height
            ):
                return math.inf
            if f["b_list_start"] and not (f["a_list_start"] or f["a_heading_like"]):
                return math.inf
            if f["a_sentence_complete"] and gap >= 0.5 and not (
                same_list or f["b_continuation_start"] or (f["a_heading_like"] and f["b_list_start"])
            ):
                return math.inf
        alignment = min(f["left_alignment_delta_norm"], f["right_alignment_delta_norm"])
        distance = 0.65 * min(gap / 3.0, 1.0) + 0.35 * min(alignment / 0.08, 1.0)
        if f["semantic_available"]:
            distance = 0.8 * distance + 0.2 * (1.0 - f["semantic_similarity"]) / 2.0
        return max(1e-8, distance)

    def prepare(self, blocks, candidates, features):
        self._labels = {block.block_id: -1 for block in blocks}
        self._distances = {}
        if not candidates:
            return
        try:
            from scipy.sparse import csr_matrix
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import sort_graph_by_row_values
        except ImportError as exc:
            raise GroupingModelUnavailable("install the grouping optional dependency") from exc
        ordered = sorted(blocks, key=lambda block: block.document_order)
        by_id = {block.block_id: block for block in ordered}
        indices = {block.block_id: index for index, block in enumerate(ordered)}
        page_bounds: dict[int, tuple[str, str]] = {}
        for block in ordered:
            first = page_bounds.get(block.page_number, (block.block_id, block.block_id))[0]
            page_bounds[block.page_number] = (first, block.block_id)
        rows = list(range(len(ordered)))
        cols = list(rows)
        distances = [0.0] * len(ordered)
        for candidate in candidates:
            a, b = by_id[candidate.block_a_id], by_id[candidate.block_b_id]
            distance = self._distance(a, b, candidate, features[candidate.pair_id], page_bounds)
            self._distances[candidate.pair_id] = distance
            if distance <= self.eps:
                i, j = indices[a.block_id], indices[b.block_id]
                rows.extend((i, j)); cols.extend((j, i)); distances.extend((distance, distance))
        graph = csr_matrix((distances, (rows, cols)), shape=(len(ordered), len(ordered)))
        graph = sort_graph_by_row_values(graph, warn_when_not_sorted=False)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed", n_jobs=1).fit_predict(graph)
        self._labels = {block.block_id: int(label) for block, label in zip(ordered, labels, strict=True)}

    def predict(self, block_a: TextBlock, block_b: TextBlock, candidate: CandidatePair,
                features: FeatureVector, *, semantic_available: bool):
        label = self._labels.get(block_a.block_id, -1)
        distance = self._distances.get(candidate.pair_id, math.inf)
        same_cluster = label >= 0 and label == self._labels.get(block_b.block_id) and distance <= self.eps
        # Affinity is deliberately not presented as a calibrated probability.
        score = (self.merge_threshold + (1 - self.merge_threshold) * (1 - distance / self.eps)) if same_cluster else 0.0
        return prediction_from_score(candidate, score, merge_threshold=self.merge_threshold,
                                     uncertain_lower=self.uncertain_lower,
                                     semantic_available=semantic_available, model_package_id=self.package_id)
