"""
tests/test_neurolinkdb.py
──────────────────────────
Phase 2 test suite for NeuroLinkDB.

Covers:
  - Page header serialisation round-trip
  - Page checksum validation and corruption detection
  - Record serialisation / deserialisation round-trip
  - Collection: insert, get, scan, delete
  - Collection: multiple pages (fill beyond one page)
  - Collection: persistence across close/reopen
  - Collection: corrupted page detection
  - NeuroLinkDB: open/close lifecycle, collection naming, drop
  - Edge cases: empty key search, large value, duplicate keys

All tests run fully offline with no external services.
"""

from __future__ import annotations

import json
import struct
import tempfile
import zlib
from pathlib import Path

import pytest

from neurochat.neurolinkdb.constants import (
    PAGE_SIZE,
    PAGE_HEADER_SIZE,
    PAGE_BODY_SIZE,
    PAGE_MAGIC,
    FORMAT_VERSION,
    PAGE_TYPE_DATA,
    NO_NEXT_PAGE,
    RECORD_HEADER_SIZE,
    MAX_RECORD_SIZE,
)
from neurochat.neurolinkdb.page import Page, PageHeader, PageCorruptedError, RecordTooLargeError
from neurochat.neurolinkdb.record import Record, serialise, deserialise
from neurochat.neurolinkdb.collection import Collection
from neurochat.neurolinkdb.database import NeuroLinkDB


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    """A temporary directory for database files."""
    return tmp_path / "testdb"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    """Open a fresh NeuroLinkDB in a temp directory."""
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def col(db: NeuroLinkDB) -> Collection:
    """A fresh collection in the temp database."""
    return db.collection("test_col")


# ─────────────────────────────────────────────────────────────────────────────
# Page header tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPageHeader:
    def test_default_header_pack_unpack(self) -> None:
        h = PageHeader(page_id=7)
        raw = h.pack()
        assert len(raw) == PAGE_HEADER_SIZE
        h2 = PageHeader.unpack(raw)
        assert h2.magic == PAGE_MAGIC
        assert h2.version == FORMAT_VERSION
        assert h2.page_id == 7
        assert h2.num_records == 0
        assert h2.free_offset == 0
        assert h2.next_page == NO_NEXT_PAGE

    def test_header_roundtrip_preserves_all_fields(self) -> None:
        h = PageHeader(
            page_id=42,
            num_records=3,
            free_offset=128,
            next_page=43,
            checksum=0xDEADBEEF,
        )
        h2 = PageHeader.unpack(h.pack())
        assert h2.page_id == 42
        assert h2.num_records == 3
        assert h2.free_offset == 128
        assert h2.next_page == 43
        assert h2.checksum == 0xDEADBEEF

    def test_unpack_raises_on_short_data(self) -> None:
        with pytest.raises(PageCorruptedError, match="too short"):
            PageHeader.unpack(b"\x00" * (PAGE_HEADER_SIZE - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Page tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPage:
    def test_new_page_size(self) -> None:
        p = Page.new(page_id=0)
        raw = p.to_bytes()
        assert len(raw) == PAGE_SIZE

    def test_page_roundtrip(self) -> None:
        p = Page.new(page_id=5)
        p.write_body_bytes(0, b"hello world")
        raw = p.to_bytes()
        p2 = Page.from_bytes(raw)
        assert p2.header.page_id == 5
        assert p2.read_body(0, 11) == b"hello world"

    def test_checksum_is_recomputed_on_serialise(self) -> None:
        p = Page.new(page_id=0)
        p.header.checksum = 0   # deliberately wrong
        raw = p.to_bytes()
        # from_bytes should not raise (checksum is recomputed)
        p2 = Page.from_bytes(raw)
        assert p2.header.checksum != 0

    def test_wrong_size_raises(self) -> None:
        with pytest.raises(PageCorruptedError, match="Expected"):
            Page.from_bytes(b"\x00" * (PAGE_SIZE - 1))

    def test_bad_magic_raises(self) -> None:
        p = Page.new(page_id=0)
        raw = bytearray(p.to_bytes())
        raw[0:4] = b"XXXX"   # corrupt magic
        # Recompute checksum so only magic is wrong
        body = bytes(raw[PAGE_HEADER_SIZE:])
        crc = zlib.crc32(body) & 0xFFFF_FFFF
        struct.pack_into("<I", raw, 24, crc)
        with pytest.raises(PageCorruptedError, match="magic"):
            Page.from_bytes(bytes(raw))

    def test_bad_checksum_raises(self) -> None:
        p = Page.new(page_id=0)
        raw = bytearray(p.to_bytes())
        raw[PAGE_HEADER_SIZE] ^= 0xFF   # flip a body byte → checksum mismatch
        with pytest.raises(PageCorruptedError, match="CRC32"):
            Page.from_bytes(bytes(raw))

    def test_validate_false_skips_checks(self) -> None:
        p = Page.new(page_id=0)
        raw = bytearray(p.to_bytes())
        raw[0:4] = b"XXXX"
        # Should not raise when validate=False
        p2 = Page.from_bytes(bytes(raw), validate=False)
        assert p2 is not None

    def test_write_beyond_body_raises(self) -> None:
        p = Page.new(page_id=0)
        with pytest.raises(ValueError, match="exceed page body"):
            p.write_body_bytes(PAGE_BODY_SIZE - 2, b"toolong!")

    def test_free_space_decreases_after_write(self) -> None:
        p = Page.new(page_id=0)
        initial = p.free_space
        p.write_body_bytes(0, b"x" * 100)
        # free_space uses free_offset, which we must update manually here
        p.header.free_offset = 100
        assert p.free_space == initial - 100


# ─────────────────────────────────────────────────────────────────────────────
# Record serialisation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordSerialisation:
    def test_simple_roundtrip(self) -> None:
        rec = Record(record_id=1, key="hello", value={"x": 42})
        raw = serialise(rec)
        rec2, consumed = deserialise(raw)
        assert consumed == len(raw)
        assert rec2.record_id == 1
        assert rec2.key == "hello"
        assert rec2.value == {"x": 42}
        assert rec2.deleted is False

    def test_deleted_flag_preserved(self) -> None:
        rec = Record(record_id=5, key="k", value="v", deleted=True)
        raw = serialise(rec)
        rec2, _ = deserialise(raw)
        assert rec2.deleted is True

    def test_non_ascii_key_and_value(self) -> None:
        rec = Record(record_id=0, key="クエリ", value={"emoji": "🧠"})
        raw = serialise(rec)
        rec2, _ = deserialise(raw)
        assert rec2.key == "クエリ"
        assert rec2.value["emoji"] == "🧠"

    def test_various_value_types(self) -> None:
        for val in [None, True, 3.14, [1, 2, 3], {"nested": {"a": 1}}]:
            rec = Record(record_id=0, key="k", value=val)
            raw = serialise(rec)
            rec2, _ = deserialise(raw)
            assert rec2.value == val

    def test_deserialise_with_offset(self) -> None:
        rec = Record(record_id=9, key="off", value="test")
        raw = serialise(rec)
        padded = b"\x00" * 20 + raw
        rec2, next_off = deserialise(padded, offset=20)
        assert rec2.key == "off"
        assert next_off == 20 + len(raw)

    def test_too_large_record_raises(self) -> None:
        big_value = "x" * MAX_RECORD_SIZE
        rec = Record(record_id=0, key="k", value=big_value)
        with pytest.raises(RecordTooLargeError):
            serialise(rec)

    def test_deserialise_short_data_raises(self) -> None:
        with pytest.raises(ValueError, match="enough bytes"):
            deserialise(b"\x00" * (RECORD_HEADER_SIZE - 1))

    def test_serialised_size_matches_rec_len(self) -> None:
        rec = Record(record_id=0, key="abc", value=123)
        raw = serialise(rec)
        # rec_len is the first 2 bytes (little-endian uint16)
        rec_len = struct.unpack_from("<H", raw, 0)[0]
        assert rec_len == len(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Collection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectionInsertGet:
    def test_insert_and_get(self, col: Collection) -> None:
        col.insert("user:1", {"name": "Alice"})
        rec = col.get("user:1")
        assert rec is not None
        assert rec.value == {"name": "Alice"}

    def test_get_returns_none_for_missing_key(self, col: Collection) -> None:
        assert col.get("ghost") is None

    def test_record_id_is_monotonic(self, col: Collection) -> None:
        r1 = col.insert("a", 1)
        r2 = col.insert("b", 2)
        r3 = col.insert("c", 3)
        assert r1.record_id < r2.record_id < r3.record_id

    def test_scan_returns_all_records(self, col: Collection) -> None:
        keys = [f"k{i}" for i in range(10)]
        for i, k in enumerate(keys):
            col.insert(k, i)
        found = [r.key for r in col.scan()]
        assert found == keys

    def test_count_matches_inserts(self, col: Collection) -> None:
        for i in range(5):
            col.insert(f"item:{i}", i)
        assert col.count() == 5

    def test_get_all_duplicates(self, col: Collection) -> None:
        col.insert("dup", "first")
        col.insert("dup", "second")
        results = col.get_all("dup")
        assert len(results) == 2
        assert results[0].value == "first"
        assert results[1].value == "second"


class TestCollectionDelete:
    def test_delete_removes_from_scan(self, col: Collection) -> None:
        col.insert("del_me", "value")
        col.delete("del_me")
        assert col.get("del_me") is None

    def test_delete_returns_count(self, col: Collection) -> None:
        col.insert("x", 1)
        col.insert("x", 2)
        count = col.delete("x")
        assert count == 2

    def test_delete_non_existent_returns_zero(self, col: Collection) -> None:
        assert col.delete("nothing") == 0

    def test_delete_only_matching_key(self, col: Collection) -> None:
        col.insert("keep", "here")
        col.insert("remove", "gone")
        col.delete("remove")
        assert col.get("keep") is not None
        assert col.get("remove") is None


class TestCollectionMultiplePages:
    def test_fills_multiple_pages(self, col: Collection) -> None:
        """
        Insert enough records to force allocation of a second page.
        Each record's body is ~PAGE_BODY_SIZE // 3 bytes to guarantee overflow.
        """
        # A value large enough that 3 records fill a page
        large_val = "A" * (PAGE_BODY_SIZE // 4)
        inserted = 0
        # Insert until we have at least 2 pages
        for i in range(30):
            col.insert(f"big:{i}", large_val)
            inserted += 1
            if col.page_count >= 2:
                break

        assert col.page_count >= 2, "Expected at least 2 pages after inserting large records"
        assert col.count() == inserted

    def test_all_records_retrievable_across_pages(self, col: Collection) -> None:
        large_val = "B" * (PAGE_BODY_SIZE // 4)
        keys = []
        for i in range(20):
            col.insert(f"page_test:{i}", large_val)
            keys.append(f"page_test:{i}")

        found = {r.key for r in col.scan()}
        assert set(keys).issubset(found)

    def test_page_count_property(self, col: Collection) -> None:
        assert col.page_count == 1   # fresh collection starts with one page


class TestCollectionPersistence:
    def test_records_survive_reopen(self, tmp_db_dir: Path) -> None:
        """Records written in one session must be readable after reopening."""
        db1 = NeuroLinkDB(tmp_db_dir)
        c1 = db1.collection("persist_test")
        c1.insert("msg:1", {"text": "hello"})
        c1.insert("msg:2", {"text": "world"})
        db1.close()

        db2 = NeuroLinkDB(tmp_db_dir)
        c2 = db2.collection("persist_test")
        assert c2.count() == 2
        r = c2.get("msg:1")
        assert r is not None
        assert r.value == {"text": "hello"}
        db2.close()

    def test_record_ids_continue_after_reopen(self, tmp_db_dir: Path) -> None:
        db1 = NeuroLinkDB(tmp_db_dir)
        c1 = db1.collection("ids")
        r1 = c1.insert("first", 1)
        db1.close()

        db2 = NeuroLinkDB(tmp_db_dir)
        c2 = db2.collection("ids")
        r2 = c2.insert("second", 2)
        assert r2.record_id > r1.record_id
        db2.close()

    def test_delete_survives_reopen(self, tmp_db_dir: Path) -> None:
        db1 = NeuroLinkDB(tmp_db_dir)
        c1 = db1.collection("del_persist")
        c1.insert("key", "val")
        c1.delete("key")
        db1.close()

        db2 = NeuroLinkDB(tmp_db_dir)
        c2 = db2.collection("del_persist")
        assert c2.get("key") is None
        db2.close()

    def test_multi_page_persistence(self, tmp_db_dir: Path) -> None:
        large_val = "C" * (PAGE_BODY_SIZE // 4)
        db1 = NeuroLinkDB(tmp_db_dir)
        c1 = db1.collection("multi_persist")
        for i in range(20):
            c1.insert(f"r:{i}", large_val)
        pages_written = c1.page_count
        count_written = c1.count()
        db1.close()

        db2 = NeuroLinkDB(tmp_db_dir)
        c2 = db2.collection("multi_persist")
        assert c2.count() == count_written
        assert c2.page_count == pages_written
        db2.close()


# ─────────────────────────────────────────────────────────────────────────────
# Corruption detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCorruptionDetection:
    def test_magic_corruption_detected(self, tmp_db_dir: Path) -> None:
        """Flip page magic bytes → PageCorruptedError when collection is opened."""
        db = NeuroLinkDB(tmp_db_dir)
        col = db.collection("corrupt_magic")
        col.insert("k", "v")
        db.close()

        nldb_path = tmp_db_dir / "corrupt_magic.nldb"
        raw = bytearray(nldb_path.read_bytes())
        # Corrupt the magic of page 0
        raw[0:4] = b"XXXX"
        # Recompute only CRC so only magic is wrong
        body = bytes(raw[PAGE_HEADER_SIZE:PAGE_SIZE])
        crc = zlib.crc32(body) & 0xFFFF_FFFF
        struct.pack_into("<I", raw, 24, crc)
        nldb_path.write_bytes(bytes(raw))

        db2 = NeuroLinkDB(tmp_db_dir)
        # _load_meta reads page 0 on open → bad magic is caught immediately
        with pytest.raises(PageCorruptedError, match="magic"):
            db2.collection("corrupt_magic")

    def test_checksum_corruption_detected(self, tmp_db_dir: Path) -> None:
        """Flip a body byte → CRC32 mismatch → PageCorruptedError on collection open."""
        db = NeuroLinkDB(tmp_db_dir)
        col = db.collection("corrupt_crc")
        col.insert("k", "v")
        db.close()

        nldb_path = tmp_db_dir / "corrupt_crc.nldb"
        raw = bytearray(nldb_path.read_bytes())
        # Flip one byte in the page body
        raw[PAGE_HEADER_SIZE] ^= 0xFF
        nldb_path.write_bytes(bytes(raw))

        db2 = NeuroLinkDB(tmp_db_dir)
        # _load_meta reads page 0 on open → bad CRC is caught immediately
        with pytest.raises(PageCorruptedError, match="CRC32"):
            db2.collection("corrupt_crc")
    def test_truncated_file_detected(self, tmp_db_dir: Path) -> None:
        """A file whose size is not a multiple of PAGE_SIZE is rejected."""
        db = NeuroLinkDB(tmp_db_dir)
        col = db.collection("truncated")
        col.insert("k", "v")
        db.close()

        nldb_path = tmp_db_dir / "truncated.nldb"
        raw = nldb_path.read_bytes()
        # Truncate by 100 bytes
        nldb_path.write_bytes(raw[:-100])

        db2 = NeuroLinkDB(tmp_db_dir)
        with pytest.raises(PageCorruptedError, match="truncated|multiple"):
            db2.collection("truncated")   # triggers _load_meta
        db2.close()


# ─────────────────────────────────────────────────────────────────────────────
# NeuroLinkDB database-level tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuroLinkDB:
    def test_creates_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "new_db"
        assert not d.exists()
        NeuroLinkDB(d).close()
        assert d.is_dir()

    def test_collection_creates_file(self, db: NeuroLinkDB, tmp_db_dir: Path) -> None:
        db.collection("alpha")
        assert (tmp_db_dir / "alpha.nldb").exists()

    def test_collection_names(self, db: NeuroLinkDB) -> None:
        db.collection("one")
        db.collection("two")
        names = set(db.collection_names())
        assert {"one", "two"}.issubset(names)

    def test_drop_collection(self, db: NeuroLinkDB, tmp_db_dir: Path) -> None:
        db.collection("to_drop")
        assert db.drop_collection("to_drop") is True
        assert not (tmp_db_dir / "to_drop.nldb").exists()

    def test_drop_missing_collection_returns_false(self, db: NeuroLinkDB) -> None:
        assert db.drop_collection("ghost") is False

    def test_invalid_collection_name_raises(self, db: NeuroLinkDB) -> None:
        with pytest.raises(ValueError, match="Invalid collection name"):
            db.collection("bad name!")

    def test_empty_name_raises(self, db: NeuroLinkDB) -> None:
        with pytest.raises(ValueError, match="empty"):
            db.collection("")

    def test_use_after_close_raises(self, db: NeuroLinkDB) -> None:
        db.close()
        with pytest.raises(ValueError, match="closed"):
            db.collection("x")

    def test_context_manager(self, tmp_db_dir: Path) -> None:
        with NeuroLinkDB(tmp_db_dir) as db:
            db.collection("ctx").insert("k", "v")
        assert not db.is_open

    def test_multiple_collections_independent(self, db: NeuroLinkDB) -> None:
        c1 = db.collection("col_a")
        c2 = db.collection("col_b")
        c1.insert("shared_key", "from_a")
        c2.insert("shared_key", "from_b")
        assert c1.get("shared_key").value == "from_a"   # type: ignore[union-attr]
        assert c2.get("shared_key").value == "from_b"   # type: ignore[union-attr]
