"""
neurochat.benchmarks.run_all
=============================
Single entry point that runs every NeuroChat benchmark and prints a combined
Markdown report.

Usage::

    python -m neurochat.benchmarks.run_all
"""

from __future__ import annotations

import sys
import time

from neurochat.benchmarks.storage_benchmark import run_storage_benchmarks, print_storage_table
from neurochat.benchmarks.vector_benchmark import run_vector_benchmarks, print_vector_table


def main() -> None:
    print("=" * 60)
    print("  NeuroChat / NeuroLinkDB — Performance Benchmarks")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Storage benchmarks
    # ------------------------------------------------------------------
    print("Running storage benchmarks…")
    t0 = time.perf_counter()
    storage_result = run_storage_benchmarks(n_records=500)
    storage_elapsed = time.perf_counter() - t0
    print_storage_table(storage_result)
    print(f"  (completed in {storage_elapsed:.1f}s)")
    print()

    # ------------------------------------------------------------------
    # Vector benchmarks
    # ------------------------------------------------------------------
    print("Running vector benchmarks…")
    t1 = time.perf_counter()
    vector_result = run_vector_benchmarks(
        n_vectors=1000,
        dim=64,
        n_queries=50,
        k_values=(1, 5, 10),
        nlist=8,
        nprobe_values=(1, 2, 4),
    )
    vector_elapsed = time.perf_counter() - t1
    print_vector_table(vector_result)
    print(f"  (completed in {vector_elapsed:.1f}s)")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = storage_elapsed + vector_elapsed
    print("=" * 60)
    print(f"  All benchmarks completed in {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

