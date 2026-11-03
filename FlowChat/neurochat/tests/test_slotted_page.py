"""
neurochat/tests/test_slotted_page.py
────────────────────────────────────
Tests for NeuroLinkDB Slotted-Page Layout, In-Place Updates, and Defragmentation (Vacuum).
"""

from __future__ import annotations

import struct
from pathlib import Path
import pytest

from neurochat.neurolinkdb.constants import (
    PAGE_SIZE,
    PAGE_BODY_SIZE,
    PAGE_HEADER_SIZE,
    FORMAT_VERSION_V1,
    FORMAT_VERSION_V2,
    SLOT_ENTRY_SIZE,
)
from neurochat.neurolinkdb.page import Page
from neurochat.neurolinkdb.record import Record, serialise, deserialise
from neurochat.neurolinkdb.collection import Collection
from neurochat.neurolinkdb.database import NeuroLinkDB


class TestSlottedPageLayout:
    """Low-level unit tests for Page slotted directory arithmetic and compaction."""

    def test_v2_page_is_slotted(self) -> None:
        p = Page.new(page_id=0, version=FORMAT_VERSION_V2)
        assert p.is_slotted is True
        assert p.header.version == FORMAT_VERSION_V2
        assert p.slot_count == 0
        assert p.header.upper_offset == PAGE_BODY_SIZE
        assert p.header.free_offset == 0
        assert p.free_space == PAGE_BODY_SIZE

    def test_v1_page_is_not_slotted(self) -> None:
        p = Page.new(page_id=0, version=FORMAT_VERSION_V1)
        assert p.is_slotted is False
        assert p.header.version == FORMAT_VERSION_V1

    def test_insert_record_allocates_from_top(self) -> None:
        p = Page.new(page_id=0)
        rec = Record(record_id=1, key="greeting", value={"msg": "hello"})
        raw = serialise(rec)

        slot_idx, payload_offset = p.insert_record(raw)
        assert slot_idx == 0
        assert payload_offset == PAGE_BODY_SIZE - len(raw)
        assert p.header.upper_offset == payload_offset
        assert p.header.free_offset == SLOT_ENTRY_SIZE
        assert p.header.num_records == 1
        assert p.slot_count == 1

        # Check slot directory
        off, ln = p.get_slot(0)
        assert off == payload_offset
        assert ln == len(raw)

    def test_multiple_records_slot_indexing(self) -> None:
        p = Page.new(page_id=0)
        records = [
            Record(record_id=i, key=f"k{i}", value=f"v{i}")
            for i in range(5)
        ]
        offsets = []
        for r in records:
            slot_idx, off = p.insert_record(serialise(r))
            offsets.append(off)

        assert p.slot_count == 5
        assert p.header.num_records == 5
        assert p.header.free_offset == 5 * SLOT_ENTRY_SIZE

        # Verify each slot points to valid record payload
        for i, r in enumerate(records):
            off, ln = p.get_slot(i)
            assert off == offsets[i]
            decoded, _ = deserialise(bytes(p.body), off)
            assert decoded.key == r.key
            assert decoded.value == r.value

    def test_delete_slot_marks_tombstone(self) -> None:
        p = Page.new(page_id=0)
        p.insert_record(serialise(Record(1, "k1", "v1")))
        p.insert_record(serialise(Record(2, "k2", "v2")))

        assert p.header.num_records == 2
        success = p.delete_slot(0)
        assert success is True
        assert p.header.num_records == 1

        # Slot 0 is marked empty/deleted
        off, ln = p.get_slot(0)
        assert off == 0 and ln == 0

        # Slot 1 is still active
        off1, ln1 = p.get_slot(1)
        assert off1 > 0 and ln1 > 0

    def test_insert_reuses_tombstoned_slot(self) -> None:
        p = Page.new(page_id=0)
        p.insert_record(serialise(Record(1, "k1", "v1")))
        p.insert_record(serialise(Record(2, "k2", "v2")))

        # Delete slot 0
        p.delete_slot(0)

        # Insert new record -> should reuse slot 0 without growing slot directory
        slot_idx, _ = p.insert_record(serialise(Record(3, "k3", "v3")))
        assert slot_idx == 0
        assert p.slot_count == 2
        assert p.header.num_records == 2

    def test_compact_defragments_space(self) -> None:
        p = Page.new(page_id=0)
        # Insert 3 records
        p.insert_record(serialise(Record(1, "k1", "payload_one")))
        p.insert_record(serialise(Record(2, "k2", "payload_two_to_delete")))
        p.insert_record(serialise(Record(3, "k3", "payload_three")))

        space_before_delete = p.free_space
        p.delete_slot(1)
        # Deletion marks slot 0,0 but contiguous free space hasn't grown yet
        assert p.free_space == space_before_delete

        # Run compaction
        reclaimed = p.compact()
        assert reclaimed > 0
        assert p.free_space > space_before_delete

        # Verify active records still intact after compaction
        remaining = list(p.iter_slots())
        assert len(remaining) == 2
        slots_found = {idx for idx, _, _ in remaining}
        assert 0 in slots_found
        assert 2 in slots_found

        r0, _ = deserialise(bytes(p.body), p.get_slot(0)[0])
        r2, _ = deserialise(bytes(p.body), p.get_slot(2)[0])
        assert r0.key == "k1"
        assert r2.key == "k3"

    def test_update_slot_in_place_when_smaller(self) -> None:
        p = Page.new(page_id=0)
        original_raw = serialise(Record(1, "k", "longer_original_value"))
        slot_idx, original_off = p.insert_record(original_raw)

        # Update with smaller value
        smaller_raw = serialise(Record(1, "k", "short"))
        updated = p.update_slot(slot_idx, smaller_raw)
        assert updated is True

        new_off, new_ln = p.get_slot(slot_idx)
        # Should stay at the same offset in-place
        assert new_off == original_off
        assert new_ln == len(smaller_raw)

        rec, _ = deserialise(bytes(p.body), new_off)
        assert rec.value == "short"

    def test_update_slot_allocates_when_larger(self) -> None:
        p = Page.new(page_id=0)
        original_raw = serialise(Record(1, "k", "short"))
        slot_idx, original_off = p.insert_record(original_raw)

        larger_raw = serialise(Record(1, "k", "much_longer_value_exceeding_previous_length"))
        updated = p.update_slot(slot_idx, larger_raw)
        assert updated is True

        new_off, new_ln = p.get_slot(slot_idx)
        # Should move to new lower offset (growing backward)
        assert new_off < original_off
        assert new_ln == len(larger_raw)

        rec, _ = deserialise(bytes(p.body), new_off)
        assert rec.value == "much_longer_value_exceeding_previous_length"


class TestCollectionSlottedOperations:
    """Integration tests for Collection update and vacuum operations."""

    @pytest.fixture()
    def col(self, tmp_path: Path) -> Collection:
        db = NeuroLinkDB(tmp_path / "test_db")
        return db.collection("items")

    def test_collection_update_existing_record(self, col: Collection) -> None:
        col.insert("user:1", {"name": "Alice", "role": "engineer"})
        updated = col.update("user:1", {"name": "Alice", "role": "lead"})

        assert updated is not None
        assert updated.key == "user:1"
        assert updated.value == {"name": "Alice", "role": "lead"}

        # Scan and get should return the updated value
        rec = col.get("user:1")
        assert rec is not None
        assert rec.value["role"] == "lead"

    def test_collection_update_preserves_record_id(self, col: Collection) -> None:
        original = col.insert("config", {"theme": "light"})
        updated = col.update("config", {"theme": "dark"})

        assert updated is not None
        assert updated.record_id == original.record_id

    def test_collection_update_missing_key_returns_none(self, col: Collection) -> None:
        res = col.update("non_existent_key", {"val": 123})
        assert res is None

    def test_collection_vacuum_reclaims_bytes(self, col: Collection) -> None:
        # Insert 10 records
        for i in range(10):
            col.insert(f"k{i}", f"payload_{i}_" * 10)

        # Delete 4 records
        col.delete("k2")
        col.delete("k4")
        col.delete("k6")
        col.delete("k8")

        # Run vacuum
        stats = col.vacuum()
        assert stats["bytes_reclaimed"] > 0
        assert stats["pages_compacted"] >= 1

    def test_collection_vacuum_preserves_remaining_records(self, col: Collection) -> None:
        for i in range(8):
            col.insert(f"item_{i}", {"idx": i})

        col.delete("item_1")
        col.delete("item_3")
        col.delete("item_5")

        col.vacuum()

        remaining_keys = {r.key for r in col.scan()}
        expected_keys = {"item_0", "item_2", "item_4", "item_6", "item_7"}
        assert remaining_keys == expected_keys

    def test_v1_and_v2_interoperability(self, tmp_path: Path) -> None:
        """Verify collection can read legacy V1 page and write V2 pages."""
        # 1. Create a legacy V1 page directly on disk
        v1_page = Page.new(page_id=0, version=FORMAT_VERSION_V1)
        r1 = serialise(Record(0, "legacy_k", "legacy_val"))
        v1_page.write_body_bytes(0, r1)
        v1_page.header.free_offset = len(r1)
        v1_page.header.num_records = 1

        db_dir = tmp_path / "interop_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        col_path = db_dir / "legacy.nldb"
        col_path.write_bytes(v1_page.to_bytes())

        # 2. Open via Collection
        db = NeuroLinkDB(db_dir)
        col = db.collection("legacy")

        # Can scan legacy record
        rec = col.get("legacy_k")
        assert rec is not None
        assert rec.value == "legacy_val"

        # Update legacy record
        updated = col.update("legacy_k", "updated_legacy_val")
        assert updated is not None
        assert updated.value == "updated_legacy_val"

        db.close()

