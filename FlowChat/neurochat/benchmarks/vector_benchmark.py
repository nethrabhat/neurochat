"""
neurochat.benchmarks.vector_benchmark
=======================================
Compare exact k-NN vs IVFFlat approximate nearest-neighbour performance.

Methodology
-----------
1. Generate *n_vectors* random unit vectors of dimension *dim*.
2. Choose *n_queries* query vectors (a random subset of the dataset).
3. Compute ground-truth neighbours with exact brute-force search.
4. For IVFFlat, vary ``nprobe`` in {1, 2, 4} and measure both Recall@K and
   average query latency.
5. Print results as a Markdown table.

All random operations use a fixed seed (42) so results are reproducible.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from neurochat.embeddings.similarity import cosine_similarity, normalize
from neurochat.neurolinkdb.index.ivfflat import IVFFlatIndex
from neurochat.storage.vector_store import VectorRecord


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SearchBenchmarkResult:
    """Aggregated vector-search benchmark measurements.

    Attributes
    ----------
    n_vectors:
        Size of the synthetic dataset.
    dim:
        Embedding dimension.
    k:
        Number of neighbours requested per query.
    exact_latency_ms:
        Average per-query latency for exact brute-force search (milliseconds).
    ivf_results:
        Mapping from ``nprobe`` value to ``(recall_at_k, avg_latency_ms)``.
    recall_at_k:
        Convenience property — Recall@K for the highest nprobe tested.
    """

    n_vectors: int
    dim: int
    k: int
    exact_latency_ms: float
    ivf_results: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    @property
    def recall_at_k(self) -> float:
        """Recall@K for the highest nprobe tested (best IVFFlat quality)."""
        if not self.ivf_results:
            return 0.0
        best_nprobe = max(self.ivf_results)
        return self.ivf_results[best_nprobe][0]


# ---------------------------------------------------------------------------
# Pure-Python exact k-NN helper
# ---------------------------------------------------------------------------


def _exact_knn(
    query: List[float],
    dataset: List[List[float]],
    k: int,
) -> List[int]:
    """Return indices of the *k* nearest neighbours by cosine similarity."""
    scored = [
        (cosine_similarity(query, vec), idx) for idx, vec in enumerate(dataset)
    ]
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:k]]


def _random_unit_vector(dim: int, rng: random.Random) -> List[float]:
    """Generate a uniformly-distributed unit vector of dimension *dim*."""
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:  # Extremely unlikely but guard against it
        raw[0] = 1.0
        norm = 1.0
    return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_vector_benchmarks(
    n_vectors: int = 1000,
    dim: int = 64,
    n_queries: int = 50,
    k_values: Tuple[int, ...] = (1, 5, 10),
    nlist: int = 8,
    nprobe_values: Tuple[int, ...] = (1, 2, 4),
    seed: int = 42,
) -> SearchBenchmarkResult:
    """Run all vector benchmarks and return a :class:`SearchBenchmarkResult`.

    Parameters
    ----------
    n_vectors:
        Number of synthetic vectors in the dataset.
    dim:
        Embedding dimension.
    n_queries:
        Number of query vectors to benchmark.
    k_values:
        Neighbour counts to measure Recall@K for (only the last value is
        stored in the primary result; all are printed).
    nlist:
        Number of IVFFlat clusters (Voronoi cells).
    nprobe_values:
        nprobe values to sweep.
    seed:
        Random seed for reproducibility.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Generate dataset
    # ------------------------------------------------------------------
    print(f"  [vector] generating {n_vectors} unit vectors (dim={dim})…", flush=True)
    dataset: List[List[float]] = [_random_unit_vector(dim, rng) for _ in range(n_vectors)]

    # ------------------------------------------------------------------
    # 2. Choose query vectors (random subset of dataset)
    # ------------------------------------------------------------------
    query_indices = rng.sample(range(n_vectors), min(n_queries, n_vectors))
    queries = [dataset[i] for i in query_indices]

    # ------------------------------------------------------------------
    # 3. Ground-truth exact k-NN (using the largest k)
    # ------------------------------------------------------------------
    k = max(k_values)
    print(f"  [vector] computing exact k-NN ground truth (k={k})…", flush=True)

    gt_start = time.perf_counter()
    ground_truth: List[Set[int]] = []
    for q in queries:
        gt_neighbours = set(_exact_knn(q, dataset, k))
        ground_truth.append(gt_neighbours)
    gt_elapsed = time.perf_counter() - gt_start
    exact_latency_ms = (gt_elapsed / len(queries)) * 1000

    # ------------------------------------------------------------------
    # 4. Build IVFFlat index using VectorRecord objects
    # ------------------------------------------------------------------
    print(f"  [vector] building IVFFlat index (nlist={nlist})…", flush=True)
    records: List[VectorRecord] = []
    for i, vec in enumerate(dataset):
        records.append(VectorRecord(
            vector_id=str(i),
            source_id=str(i),
            vector=vec,
            model_name="benchmark",
            dimension=dim,
            metadata={},
        ))
    index = IVFFlatIndex(dimension=dim, nlist=nlist)
    index.build(records)

    # ------------------------------------------------------------------
    # 5. Sweep nprobe values
    # ------------------------------------------------------------------
    ivf_results: Dict[int, Tuple[float, float]] = {}

    for nprobe in nprobe_values:
        print(f"  [vector] IVFFlat nprobe={nprobe}…", flush=True)
        total_recall = 0.0
        nprobe_start = time.perf_counter()

        for qi, q in enumerate(queries):
            results = index.search(q, top_k=k, nprobe=nprobe)
            result_set = set(int(r.vector_id) for r in results)
            hits = len(result_set & ground_truth[qi])
            total_recall += hits / k

        nprobe_elapsed = time.perf_counter() - nprobe_start
        avg_recall = total_recall / len(queries)
        avg_latency_ms = (nprobe_elapsed / len(queries)) * 1000
        ivf_results[nprobe] = (avg_recall, avg_latency_ms)

    return SearchBenchmarkResult(
        n_vectors=n_vectors,
        dim=dim,
        k=k,
        exact_latency_ms=exact_latency_ms,
        ivf_results=ivf_results,
    )


def print_vector_table(result: SearchBenchmarkResult) -> None:
    """Pretty-print benchmark results as a Markdown table."""
    print()
    print("## Vector Search Benchmarks")
    print()
    print(
        f"Dataset: {result.n_vectors} vectors, dim={result.dim}, "
        f"k={result.k}, n_queries=50"
    )
    print()
    print(
        "| Method              | Recall@K | Avg Latency (ms) |"
    )
    print(
        "|:--------------------|:--------:|----------------:|"
    )
    print(
        f"| Exact brute-force   |  1.0000  | "
        f"{result.exact_latency_ms:>15.3f} |"
    )
    for nprobe, (recall, latency_ms) in sorted(result.ivf_results.items()):
        label = f"IVFFlat nprobe={nprobe}"
        print(
            f"| {label:<19} | {recall:>8.4f} | {latency_ms:>15.3f} |"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running vector benchmarks…")
    result = run_vector_benchmarks()
    print_vector_table(result)
