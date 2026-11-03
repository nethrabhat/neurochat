"""
tests/test_buffer_pool_wal.py
─────────────────────────────
Phase 7 test suite: Buffer Pool Manager, LRU Eviction, Write-Ahead Logging (WAL),
and Crash Recovery.

Covers:
  - In-memory BufferPoolManager frame caching and hit-ratio tracking
  - Dirty page tracking and unpinning
  - LRU (Least Recently Used) page eviction policy
  - Automatic dirty-page writeback on eviction
  - Pin-count protection against eviction (BufferPoolFullError)
  - WriteAheadLog binary serialization, monotonic LSNs, and fsync durability
  - Redo crash recovery replaying committed transactions from WAL
  - Resilient recovery handling interrupted / torn writes and corrupted log tails
  - End-to-end Collection integration with both buffer pool and WAL

All tests run fully offline without external networks or database systems.
"""

from __future__ import annotations

import struct
from pathlib import Path
import pytest

from neurochat.neurolinkdb.constants import PAGE_SIZE, PAGE_BODY_SIZE
from neurochat.neurolinkdb.page import Page
from neurochat.neurolinkdb.buffer_pool import (
    BufferPoolManager,
    BufferPoolFullError,
    Frame,
)
from neurochat.neurolinkdb.wal import (
    WriteAheadLog,
    LogRecordType,
    WAL_MAGIC,
    WAL_HEADER_SIZE,
)
from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.neurolinkdb.collection import Collection


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    return tmp_path / "reliability_test_db"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def mock_disk(tmp_path: Path) -> tuple[dict[int, Page], BufferPoolManager]:
    """In-memory simulated disk table for isolating BufferPoolManager tests."""
    disk_storage: dict[int, Page] = {}

    def read_fn(page_id: int) -> Page:
        if page_id not in disk_storage:
            disk_storage[page_id] = Page.new(page_id=page_id)
        # Return a copy to simulate reading from disk
        return Page.from_bytes(disk_storage[page_id].to_bytes())

    def write_fn(page: Page) -> None:
        disk_storage[page.header.page_id] = Page.from_bytes(page.to_bytes())

    bpm = BufferPoolManager(read_fn, write_fn, capacity=3)
    return disk_storage, bpm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Buffer Pool Manager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBufferPoolManager:
    def test_fetch_and_cache_hit(
        self, mock_disk: tuple[dict[int, Page], BufferPoolManager]
    ) -> None:
        _, bpm = mock_disk

        # First access -> Cache Miss
        p0 = bpm.fetch_page(0)
        assert bpm.cache_misses == 1
        assert bpm.cache_hits == 0
        assert bpm.is_cached(0)

        bpm.unpin_page(0)

        # Second access -> Cache Hit
        p0_again = bpm.fetch_page(0)
        assert bpm.cache_misses == 1
        assert bpm.cache_hits == 1
        assert bpm.hit_ratio == 0.5
        bpm.unpin_page(0)

    def test_dirty_page_tracking_and_flush(
        self, mock_disk: tuple[dict[int, Page], BufferPoolManager]
    ) -> None:
        disk, bpm = mock_disk

        p = bpm.fetch_page(1)
        p.write_body_bytes(0, b"dirty data")
        bpm.unpin_page(1, is_dirty=True)

        assert bpm.get_dirty_pages() == [1]

        # Flush dirty page
        bpm.flush_page(1)
        assert bpm.get_dirty_pages() == []
        # Verify written to simulated disk
        assert disk[1].read_body(0, 10) == b"dirty data"

    def test_lru_eviction_order(
        self, mock_disk: tuple[dict[int, Page], BufferPoolManager]
    ) -> None:
        """With capacity 3, fetching a 4th unpinned page evicts the LRU page."""
        _, bpm = mock_disk
        assert bpm.capacity == 3

        # Fill capacity with pages 0, 1, 2
        bpm.fetch_page(0)
        bpm.unpin_page(0)

        bpm.fetch_page(1)
        bpm.unpin_page(1)

        bpm.fetch_page(2)
        bpm.unpin_page(2)

        assert set(bpm.page_table.keys()) == {0, 1, 2}

        # Fetching page 3 must evict page 0 (least recently accessed)
        bpm.fetch_page(3)
        bpm.unpin_page(3)

        assert bpm.evictions == 1
        assert not bpm.is_cached(0)
        assert set(bpm.page_table.keys()) == {1, 2, 3}

    def test_dirty_page_flushed_on_eviction(
        self, mock_disk: tuple[dict[int, Page], BufferPoolManager]
    ) -> None:
        disk, bpm = mock_disk

        # Fetch page 0 and modify it
        p0 = bpm.fetch_page(0)
        p0.write_body_bytes(0, b"unflushed on disk")
        bpm.unpin_page(0, is_dirty=True)

        # Fetch pages 1, 2, 3 to force page 0 out of capacity 3
        for pid in (1, 2, 3):
            bpm.fetch_page(pid)
            bpm.unpin_page(pid)

        # Page 0 was evicted, but must have been flushed to disk first
        assert not bpm.is_cached(0)
        assert disk[0].read_body(0, 17) == b"unflushed on disk"

    def test_pin_count_prevents_eviction(
        self, mock_disk: tuple[dict[int, Page], BufferPoolManager]
    ) -> None:
        _, bpm = mock_disk

        # Pin all 3 pages (do not call unpin)
        bpm.fetch_page(0)
        bpm.fetch_page(1)
        bpm.fetch_page(2)

        # Attempting to fetch a 4th page when all are pinned raises BufferPoolFullError
        with pytest.raises(BufferPoolFullError, match="currently pinned"):
            bpm.fetch_page(3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Write-Ahead Log (WAL) Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteAheadLog:
    def test_init_and_magic_header(self, tmp_path: Path) -> None:
        wal_file = tmp_path / "test.wal"
        wal = WriteAheadLog(wal_file)
        assert wal_file.exists()
        assert wal_file.read_bytes()[:4] == WAL_MAGIC

    def test_append_monotonic_lsn(self, tmp_path: Path) -> None:
        wal = WriteAheadLog(tmp_path / "lsn.wal")
        lsn1 = wal.append(LogRecordType.PAGE_WRITE, page_id=1, payload=b"payload_1")
        lsn2 = wal.append(LogRecordType.PAGE_WRITE, page_id=2, payload=b"payload_2")
        assert lsn1 == 1
        assert lsn2 == 2

    def test_checkpoint_and_truncate(self, tmp_path: Path) -> None:
        wal_file = tmp_path / "cp.wal"
        wal = WriteAheadLog(wal_file)
        wal.append(LogRecordType.PAGE_WRITE, page_id=0, payload=b"data")
        assert wal_file.stat().st_size > len(WAL_MAGIC)

        wal.checkpoint()
        wal.truncate()
        # Truncation leaves clean file with just WAL_MAGIC
        assert wal_file.stat().st_size == len(WAL_MAGIC)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Crash Recovery and Torn-Write Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrashRecovery:
    def test_redo_recovery_restores_unflushed_pages(self, tmp_path: Path) -> None:
        """Simulate crash where WAL was flushed but collection file was not updated."""
        col_file = tmp_path / "crash_test.nldb"
        wal_file = tmp_path / "crash_test.wal"

        # Initialize blank collection file
        p0 = Page.new(page_id=0)
        col_file.write_bytes(p0.to_bytes())

        # Simulate transactions writing to WAL
        wal = WriteAheadLog(wal_file)
        modified_p0 = Page.new(page_id=0)
        modified_p0.write_body_bytes(0, b"RECOVERED_PAGE_ZERO")
        wal.log_page_write(0, modified_p0.to_bytes())

        # Page 1 allocated and written to WAL
        p1 = Page.new(page_id=1)
        p1.write_body_bytes(0, b"RECOVERED_PAGE_ONE")
        wal.log_page_write(1, p1.to_bytes())

        # Collection file on disk still contains only original blank p0
        assert Page.from_bytes(col_file.read_bytes()[:PAGE_SIZE]).read_body(0, 19) != b"RECOVERED_PAGE_ZERO"

        # Execute recovery
        applied_pages: dict[int, bytes] = {}

        def apply_fn(pid: int, raw: bytes) -> None:
            applied_pages[pid] = raw

        replayed = wal.recover(apply_fn)
        assert replayed == 2
        assert Page.from_bytes(applied_pages[0]).read_body(0, 19) == b"RECOVERED_PAGE_ZERO"
        assert Page.from_bytes(applied_pages[1]).read_body(0, 18) == b"RECOVERED_PAGE_ONE"

    def test_recovery_stops_at_interrupted_torn_write(self, tmp_path: Path) -> None:
        """Simulate power loss mid-write creating a partial record at the end of the WAL."""
        wal_file = tmp_path / "torn.wal"
        wal = WriteAheadLog(wal_file)

        # Write valid record
        p0 = Page.new(page_id=0)
        p0.write_body_bytes(0, b"VALID_TX")
        wal.log_page_write(0, p0.to_bytes())

        # Simulate torn write: write only 10 bytes of a partial header
        with wal_file.open("ab") as fh:
            fh.write(b"TORN_PARTIAL_HEADER")

        applied: list[int] = []
        # Recovery must safely process the valid record and stop at the torn boundary
        replayed = wal.recover(lambda pid, raw: applied.append(pid))
        assert replayed == 1
        assert applied == [0]

    def test_recovery_stops_at_corrupted_checksum(self, tmp_path: Path) -> None:
        """Simulate bit-flip in WAL payload: checksum mismatch halts recovery cleanly."""
        wal_file = tmp_path / "corrupt_crc.wal"
        wal = WriteAheadLog(wal_file)

        # Write valid record 1
        p0 = Page.new(page_id=0)
        p0.write_body_bytes(0, b"FIRST_VALID")
        wal.log_page_write(0, p0.to_bytes())

        # Write record 2 then corrupt its payload
        p1 = Page.new(page_id=1)
        p1.write_body_bytes(0, b"SECOND_TO_CORRUPT")
        wal.log_page_write(1, p1.to_bytes())

        # Flip byte in record 2 payload
        raw_wal = bytearray(wal_file.read_bytes())
        # Corrupt near the end of the file
        raw_wal[-10] ^= 0xFF
        wal_file.write_bytes(bytes(raw_wal))

        applied: list[int] = []
        replayed = wal.recover(lambda pid, raw: applied.append(pid))
        # First record must recover successfully; corrupted second record is rejected
        assert replayed == 1
        assert applied == [0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Collection Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectionReliabilityIntegration:
    def test_collection_with_buffer_pool_and_wal(
        self, db: NeuroLinkDB
    ) -> None:
        col = db.collection(
            "reliable_col",
            use_wal=True,
            use_buffer_pool=True,
            buffer_pool_capacity=4,
        )

        assert col.wal is not None
        assert col.buffer_pool is not None

        # Insert multiple records
        for i in range(15):
            col.insert(f"k_{i}", f"val_{i}")

        assert col.count() == 15
        assert col.buffer_pool.cache_hits > 0

        # Flushing checkpoints WAL
        col.flush()
        db.close()

    def test_collection_reopen_with_wal(self, tmp_db_dir: Path) -> None:
        # Session 1: insert with WAL enabled
        db1 = NeuroLinkDB(tmp_db_dir)
        c1 = db1.collection("wal_persist", use_wal=True, use_buffer_pool=True)
        c1.insert("alpha", "first_value")
        c1.insert("beta", "second_value")
        db1.close()

        # Session 2: reopen and verify automatic recovery and record accessibility
        db2 = NeuroLinkDB(tmp_db_dir)
        c2 = db2.collection("wal_persist", use_wal=True, use_buffer_pool=True)
        assert c2.count() == 2
        assert c2.get("alpha").value == "first_value"
        assert c2.get("beta").value == "second_value"
        db2.close()
