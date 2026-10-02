import pytest
from scripts.benchmark_grouping import benchmark


def test_benchmark_measures_pipeline_and_candidate_recall():
    result = benchmark(100,repeats=2)
    assert result['candidate_pairs'] > 0 and result['groups'] > 0
    for stage in ('normalization_time','block_time','candidate_time','feature_time','classifier_time','graph_time','group_time','projection_time','total_time'):
        assert result['timings_ms'][stage] >= 0
    assert 'learned_overhead_ms' not in result
    assert 'p95_resolver_ms' not in result
    assert result['candidate_diagnostics']['adjacent_candidate_recall'] == 1


def test_warm_mock_embedding_benchmark_reports_real_cache_activity():
    result = benchmark(20,semantics='mock',warm_cache=True,repeats=2)
    assert result['embedding']['cache_hits'] == 20
    assert result['embedding']['batch_count'] == 0


@pytest.mark.parametrize('blocks', [0,-1])
def test_benchmark_rejects_empty_workloads(blocks):
    with pytest.raises(ValueError):
        benchmark(blocks)
