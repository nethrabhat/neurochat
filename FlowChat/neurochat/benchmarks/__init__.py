"""
NeuroChat benchmarks package.

Run all benchmarks via::

    python -m neurochat.benchmarks.run_all

Or individually::

    python -m neurochat.benchmarks.storage_benchmark
    python -m neurochat.benchmarks.vector_benchmark
"""

from neurochat.benchmarks.storage_benchmark import StorageBenchmarkResult, run_storage_benchmarks
from neurochat.benchmarks.vector_benchmark import SearchBenchmarkResult, run_vector_benchmarks

__all__ = [
    "StorageBenchmarkResult",
    "SearchBenchmarkResult",
    "run_storage_benchmarks",
    "run_vector_benchmarks",
]

