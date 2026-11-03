"""
Tests for Phase 8b — benchmark modules.

These are smoke tests: they verify that each benchmark function runs without
error and returns a result object with sane values.  We use a very small
dataset so they complete in a few seconds.
"""

from __future__ import annotations

import pytest

from neurochat.benchmarks.storage_benchmark import (
    StorageBenchmarkResult,
    run_storage_benchmarks,
)
from neurochat.benchmarks.vector_benchmark import (
    SearchBenchmarkResult,
    run_vector_benchmarks,
)


# ---------------------------------------------------------------------------
# Storage benchmarks
# ---------------------------------------------------------------------------


class TestStorageBenchmarks:
    @pytest.fixture(scope="class")
    def result(self):
        """Run storage benchmarks once for the whole class (small dataset)."""
        return run_storage_benchmarks(n_records=50)

    def test_returns_correct_type(self, result):
        assert isinstance(result, StorageBenchmarkResult)

    def test_n_records_matches(self, result):
        assert result.n_records == 50

    def test_insert_plain_positive(self, result):
        assert result.insert_plain_rps > 0

    def test_insert_bpm_positive(self, result):
        assert result.insert_bpm_rps > 0

    def test_scan_plain_non_negative(self, result):
        assert result.scan_plain_ms >= 0

    def test_scan_bpm_non_negative(self, result):
        assert result.scan_bpm_ms >= 0


# ---------------------------------------------------------------------------
# Vector benchmarks
# ---------------------------------------------------------------------------


class TestVectorBenchmarks:
    @pytest.fixture(scope="class")
    def result(self):
        """Run vector benchmarks once for the whole class (tiny dataset)."""
        return run_vector_benchmarks(
            n_vectors=100,
            dim=16,
            n_queries=10,
            k_values=(1, 3),
            nlist=4,
            nprobe_values=(1, 2),
            seed=0,
        )

    def test_returns_correct_type(self, result):
        assert isinstance(result, SearchBenchmarkResult)

    def test_exact_latency_non_negative(self, result):
        assert result.exact_latency_ms >= 0.0

    def test_recall_at_k_in_valid_range(self, result):
        recall = result.recall_at_k
        assert 0.0 <= recall <= 1.0

    def test_ivf_results_for_each_nprobe(self, result):
        assert set(result.ivf_results.keys()) == {1, 2}

    def test_higher_nprobe_not_worse_recall(self, result):
        """Recall should be >= nprobe=1 when nprobe=2."""
        r1 = result.ivf_results[1][0]
        r2 = result.ivf_results[2][0]
        # Allow small floating-point noise: r2 should be >= r1 or equal
        assert r2 >= r1 - 1e-9

    def test_latency_values_non_negative(self, result):
        for nprobe, (recall, latency_ms) in result.ivf_results.items():
            assert latency_ms >= 0.0, f"nprobe={nprobe} has negative latency"

    def test_full_nprobe_achieves_high_recall(self):
        """nprobe == nlist should achieve perfect (or near-perfect) recall."""
        result = run_vector_benchmarks(
            n_vectors=100,
            dim=8,
            n_queries=10,
            k_values=(1,),
            nlist=4,
            nprobe_values=(4,),  # nprobe == nlist → exhaustive over all clusters
            seed=7,
        )
        recall = result.ivf_results[4][0]
        assert recall >= 0.9, f"Expected recall >= 0.9 with nprobe==nlist, got {recall:.4f}"

