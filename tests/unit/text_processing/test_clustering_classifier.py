from dataclasses import replace

import pytest

pytest.importorskip("sklearn")

from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.text_processing.classifiers.clustering_classifier import ClusteringRelationshipClassifier
from tests.fixtures.grouping.factories import make_block


def cluster(blocks, **options):
    return resolve_paragraphs(blocks, ClusteringRelationshipClassifier(**options))


def test_clustering_joins_continuations_and_keeps_distant_paragraphs_separate():
    blocks = [make_block("a", "An unfinished paragraph", ordinal=0),
              make_block("b", "continues on the next line.", ordinal=1, y1=140, y2=170),
              make_block("c", "A separate paragraph.", ordinal=2, y1=400, y2=430)]
    result = cluster(blocks)
    assert [[m.block_id for m in group.members] for group in result.groups] == [["a", "b"], ["c"]]
    assert result.groups[0].confidence_method == "minimum_clustering_affinity"
    assert result.groups[0].metadata["algorithm"] == "dbscan"


def test_clustering_noise_points_do_not_merge_and_density_is_configurable():
    blocks = [make_block("a", "unfinished", ordinal=0),
              make_block("b", "continuation", ordinal=1, y1=140, y2=170)]
    assert len(cluster(blocks, min_samples=3).groups) == 2
    assert len(cluster(blocks).groups) == 1
    assert len(cluster(blocks, eps=0.01).groups) == 2


def test_clustering_respects_columns_completed_paragraphs_and_new_headings():
    a = make_block("a", "A complete paragraph.", ordinal=0)
    b = make_block("b", "Another paragraph.", ordinal=1, y1=150, y2=180)
    assert len(cluster([a, b]).groups) == 2
    a = replace(a, original_text="unfinished", normalized_text="unfinished")
    b = replace(b, column_id="other")
    assert len(cluster([a, b]).groups) == 2
    b = replace(b, column_id="main", block_type="heading")
    assert len(cluster([a, b]).groups) == 2


def test_clustering_does_not_bridge_across_an_intervening_boundary():
    blocks = [make_block("a", "unfinished", ordinal=0),
              replace(make_block("title", "A new heading", ordinal=1, y1=135, y2=165), block_type="heading"),
              make_block("b", "continuation", ordinal=2, y1=170, y2=200)]
    result = cluster(blocks)
    assert all(not {"a", "b"}.issubset({m.block_id for m in group.members}) for group in result.groups)


def test_clustering_preserves_lists_and_is_stable_under_input_reordering():
    blocks = [make_block("a", "1. First item", ordinal=0),
              make_block("b", "continued detail", ordinal=1, y1=140, y2=170),
              make_block("c", "2. Second item", ordinal=2, y1=180, y2=210)]
    first = cluster(blocks)
    assert len(first.groups) == 1 and first.groups[0].group_type.value == "list"
    assert [g.group_id for g in first.groups] == [g.group_id for g in cluster(list(reversed(blocks))).groups]


def test_clustering_cross_page_links_require_boundary_continuation():
    a = make_block("a", "An unfinished paragraph", y1=1320, y2=1350)
    b = make_block("b", "continues on the next page.", page_number=2, y1=40, y2=70)
    result = cluster([a, b])
    assert len(result.groups) == 1 and len(result.groups[0].page_spans) == 2
    complete = replace(a, original_text="Complete paragraph.", normalized_text="Complete paragraph.")
    assert len(cluster([complete, b]).groups) == 2
    middle = make_block("middle", "unfinished", y1=500, y2=530)
    assert len(cluster([middle, b]).groups) == 2


def test_classifier_can_be_reused_without_carrying_clusters_between_documents():
    classifier = ClusteringRelationshipClassifier()
    blocks = [make_block("a", "unfinished", ordinal=0),
              make_block("b", "continuation", ordinal=1, y1=140, y2=170)]
    assert len(resolve_paragraphs(blocks, classifier).groups) == 1
    assert len(resolve_paragraphs([blocks[0]], classifier).groups) == 1
    assert classifier._labels == {"a": -1}
