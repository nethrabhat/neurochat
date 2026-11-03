"""
neurolinkdb/index/evaluator.py
──────────────────────────────
Evaluation and benchmarking tools for vector search.

Measures Recall@K, latency comparison between exact and approximate IVFFlat
searches, build time, and storage overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from neurochat.storage.vector_store import VectorSearchResult, VectorStore


@dataclass
class SearchBenchmarkResult:
    """Benchmark metrics comparing exact vs approximate index search."""

    top_k: int
    nprobe: int
    recall_at_k: float
    exact_latency_ms: float
    approx_latency_ms: float
    speedup: float


def compute_recall_at_k(
    exact_results: Sequence[VectorSearchResult],
    approx_results: Sequence[VectorSearchResult],
    k: int,
) -> float:
    """
    Compute Recall@K metric:
        Recall@K = |Exact_TopK ∩ Approx_TopK| / K

    Parameters
    ----------
    exact_results:
        Ranked results from exhaustive exact k-NN.
    approx_results:
        Ranked results from approximate index search.
    k:
        Top-k boundary to evaluate.

    Returns
    -------
    float
        Fraction of true nearest neighbors retrieved in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    target_k = min(k, len(exact_results))
    if target_k == 0:
        return 1.0  # Both empty, vacuous truth

    exact_ids = {r.source_id for r in exact_results[:target_k]}
    approx_ids = {r.source_id for r in approx_results[:target_k]}

    intersection = exact_ids.intersection(approx_ids)
    return len(intersection) / float(target_k)


def evaluate_index_performance(
    vector_store: VectorStore,
    test_queries: Sequence[Sequence[float]],
    *,
    top_k: int = 5,
    nprobe: int = 1,
) -> SearchBenchmarkResult:
    """
    Benchmark exact vs IVFFlat approximate search on a sample query set.
    """
    if not test_queries:
        raise ValueError("test_queries must not be empty.")

    exact_latencies: List[float] = []
    approx_latencies: List[float] = []
    recalls: List[float] = []

    for q in test_queries:
        # Measure Exact search
        t0 = time.perf_counter()
        exact_res = vector_store.search(q, top_k=top_k, use_index=False)
        exact_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Measure IVFFlat search
        t1 = time.perf_counter()
        approx_res = vector_store.search(
            q, top_k=top_k, use_index=True, nprobe=nprobe
        )
        approx_latencies.append((time.perf_counter() - t1) * 1000.0)

        recall = compute_recall_at_k(exact_res, approx_res, top_k)
        recalls.append(recall)

    mean_exact = sum(exact_latencies) / len(exact_latencies)
    mean_approx = sum(approx_latencies) / len(approx_latencies)
    mean_recall = sum(recalls) / len(recalls)
    speedup = (mean_exact / mean_approx) if mean_approx > 0.0 else 1.0

    return SearchBenchmarkResult(
        top_k=top_k,
        nprobe=nprobe,
        recall_at_k=mean_recall,
        exact_latency_ms=mean_exact,
        approx_latency_ms=mean_approx,
        speedup=speedup,
    )

