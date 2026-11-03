"""
neurochat.benchmarks.storage_benchmark
=======================================
Measure NeuroLinkDB Collection insert / scan throughput.

Benchmarks
----------
1. Insert throughput  — plain Collection (no buffer pool)
2. Insert throughput  — Collection with BufferPoolManager enabled
3. Scan latency       — plain Collection after N inserts
4. Scan latency       — Collection with BufferPoolManager after N inserts

All measurements are wall-clock time via :func:`time.perf_counter`.
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from neurochat.neurolinkdb.database import NeuroLinkDB


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StorageBenchmarkResult:
    """Aggregated storage benchmark measurements.

    Attributes
    ----------
    n_records:
        Number of records inserted in each run.
    insert_plain_rps:
        Insert throughput without buffer pool (records / second).
    insert_bpm_rps:
        Insert throughput with buffer pool (records / second).
    scan_plain_ms:
        Wall-clock time to scan all records without buffer pool (milliseconds).
    scan_bpm_ms:
        Wall-clock time to scan all records with buffer pool (milliseconds).
    """

    n_records: int
    insert_plain_rps: float
    insert_bpm_rps: float
    scan_plain_ms: float
    scan_bpm_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(i: int) -> dict:
    return {
        "index": i,
        "text": f"Benchmark record number {i} containing some sample payload text.",
        "tags": ["bench", "test"],
    }


def _bench_insert(db_dir: Path, n: int, *, use_buffer_pool: bool) -> float:
    """Return insert throughput in records/second."""
    db = NeuroLinkDB(str(db_dir))
    col = db.collection("bench_insert", use_buffer_pool=use_buffer_pool)

    start = time.perf_counter()
    for i in range(n):
        col.insert(str(uuid.uuid4()), _make_payload(i))
    elapsed = time.perf_counter() - start

    if use_buffer_pool:
        col.flush()
    db.close()
    return n / elapsed if elapsed > 0 else float("inf")


def _bench_scan(db_dir: Path, n: int, *, use_buffer_pool: bool) -> float:
    """Populate a collection with *n* records then return scan latency in ms."""
    db = NeuroLinkDB(str(db_dir))
    col = db.collection("bench_scan", use_buffer_pool=use_buffer_pool)

    # Populate
    for i in range(n):
        col.insert(str(uuid.uuid4()), _make_payload(i))
    if use_buffer_pool:
        col.flush()
    db.close()

    # Re-open and measure scan
    db2 = NeuroLinkDB(str(db_dir))
    col2 = db2.collection("bench_scan", use_buffer_pool=use_buffer_pool)
    start = time.perf_counter()
    results = list(col2.scan())
    elapsed_ms = (time.perf_counter() - start) * 1000
    db2.close()

    assert len(results) == n, f"Expected {n} records, got {len(results)}"
    return elapsed_ms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_storage_benchmarks(n_records: int = 500) -> StorageBenchmarkResult:
    """Run all storage benchmarks and return a :class:`StorageBenchmarkResult`.

    Parameters
    ----------
    n_records:
        Number of records to use in each sub-benchmark (default 500).
        Larger values give more stable measurements but take longer.
    """
    with tempfile.TemporaryDirectory(prefix="nldb_bench_") as tmp:
        base = Path(tmp)

        print(f"  [storage] insert plain ({n_records} records)…", flush=True)
        insert_plain = _bench_insert(base / "plain_insert", n_records, use_buffer_pool=False)

        print(f"  [storage] insert with buffer pool ({n_records} records)…", flush=True)
        insert_bpm = _bench_insert(base / "bpm_insert", n_records, use_buffer_pool=True)

        print(f"  [storage] scan plain ({n_records} records)…", flush=True)
        scan_plain = _bench_scan(base / "plain_scan", n_records, use_buffer_pool=False)

        print(f"  [storage] scan with buffer pool ({n_records} records)…", flush=True)
        scan_bpm = _bench_scan(base / "bpm_scan", n_records, use_buffer_pool=True)

    return StorageBenchmarkResult(
        n_records=n_records,
        insert_plain_rps=insert_plain,
        insert_bpm_rps=insert_bpm,
        scan_plain_ms=scan_plain,
        scan_bpm_ms=scan_bpm,
    )


def print_storage_table(result: StorageBenchmarkResult) -> None:
    """Pretty-print benchmark results as a Markdown table."""
    print()
    print("## Storage Benchmarks")
    print()
    print(f"Dataset: {result.n_records} records")
    print()
    print(f"| Metric                      | Plain Collection | + Buffer Pool |")
    print(f"|:----------------------------|----------------:|-------------:|")
    print(
        f"| Insert throughput (rec/s)   | "
        f"{result.insert_plain_rps:>14,.0f} | "
        f"{result.insert_bpm_rps:>12,.0f} |"
    )
    print(
        f"| Scan latency (ms)           | "
        f"{result.scan_plain_ms:>14.2f} | "
        f"{result.scan_bpm_ms:>12.2f} |"
    )
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running storage benchmarks…")
    result = run_storage_benchmarks()
    print_storage_table(result)

