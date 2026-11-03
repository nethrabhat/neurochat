"""
neurolinkdb/page.py
────────────────────
Binary page layout for NeuroLinkDB.

Page layout (4096 bytes total):

  ┌─────────────────────────────────────── offset 0 ───┐
  │  PAGE HEADER  (32 bytes)                            │
  │   [0:4]   magic        b'NLDB'                      │
  │   [4:6]   version      uint16  (currently 1)        │
  │   [6:8]   page_type    uint16  (0=free,1=data,2=ov) │
  │   [8:12]  page_id      uint32                       │
  │  [12:16]  num_records  uint32                       │
  │  [16:20]  free_offset  uint32  (into body, 0-based) │
  │  [20:24]  next_page    uint32  (0xFFFFFFFF = none)  │
  │  [24:28]  checksum     uint32  CRC32 of body bytes  │
  │  [28:32]  reserved     uint32  (zeroed)             │
  ├─────────────────────────────────────── offset 32 ───┤
  │  PAGE BODY   (4064 bytes)                           │
  │  Variable-length record slots packed sequentially.  │
  └─────────────────────────────────────────────────────┘

All multi-byte integers are little-endian.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

from neurochat.neurolinkdb.constants import (
    PAGE_SIZE,
    PAGE_HEADER_SIZE,
    PAGE_BODY_SIZE,
    PAGE_MAGIC,
    FORMAT_VERSION,
    FORMAT_VERSION_V1,
    FORMAT_VERSION_V2,
    SLOT_ENTRY_SIZE,
    PAGE_TYPE_DATA,
    NO_NEXT_PAGE,
)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class PageCorruptedError(Exception):
    """Raised when a page fails magic-byte or checksum validation."""


class RecordTooLargeError(Exception):
    """Raised when a serialised record exceeds MAX_RECORD_SIZE."""


# ── Header struct: 32 bytes, little-endian ────────────────────────────────────
# Format string breakdown:
#   4s  magic        (4 bytes, raw bytes)
#   H   version      (2 bytes, uint16)
#   H   page_type    (2 bytes, uint16)
#   I   page_id      (4 bytes, uint32)
#   I   num_records  (4 bytes, uint32)
#   I   free_offset  (4 bytes, uint32)
#   I   next_page    (4 bytes, uint32)
#   I   checksum     (4 bytes, uint32)
#   I   reserved     (4 bytes, uint32)
_HEADER_STRUCT = struct.Struct("<4sHHIIIIII")
assert _HEADER_STRUCT.size == PAGE_HEADER_SIZE, (
    f"Header struct size mismatch: {_HEADER_STRUCT.size} != {PAGE_HEADER_SIZE}"
)


@dataclass
class PageHeader:
    """Parsed representation of the 32-byte page header."""

    magic: bytes = PAGE_MAGIC
    version: int = FORMAT_VERSION
    page_type: int = PAGE_TYPE_DATA
    page_id: int = 0
    num_records: int = 0
    free_offset: int = 0          # in V1: next append offset; in V2: end of slot directory
    next_page: int = NO_NEXT_PAGE
    checksum: int = 0
    reserved: int = PAGE_BODY_SIZE  # in V1: 0; in V2: upper_offset (payload start growing backward)

    # ── V2 slotted-page helpers ───────────────────────────────────────────────

    @property
    def upper_offset(self) -> int:
        """In V2 slotted pages, start of payload space growing backward."""
        return self.reserved if self.reserved != 0 else PAGE_BODY_SIZE

    @upper_offset.setter
    def upper_offset(self, val: int) -> None:
        self.reserved = val

    # ── serialisation ─────────────────────────────────────────────────────────

    def pack(self) -> bytes:
        """Serialise header to exactly PAGE_HEADER_SIZE bytes."""
        return _HEADER_STRUCT.pack(
            self.magic,
            self.version,
            self.page_type,
            self.page_id,
            self.num_records,
            self.free_offset,
            self.next_page,
            self.checksum,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "PageHeader":
        """Deserialise header from the first PAGE_HEADER_SIZE bytes of *data*."""
        if len(data) < PAGE_HEADER_SIZE:
            raise PageCorruptedError(
                f"Header data too short: {len(data)} < {PAGE_HEADER_SIZE}"
            )
        (
            magic, version, page_type, page_id,
            num_records, free_offset, next_page,
            checksum, reserved,
        ) = _HEADER_STRUCT.unpack(data[:PAGE_HEADER_SIZE])
        return cls(
            magic=magic,
            version=version,
            page_type=page_type,
            page_id=page_id,
            num_records=num_records,
            free_offset=free_offset,
            next_page=next_page,
            checksum=checksum,
            reserved=reserved,
        )


# ── Slot struct: 4 bytes, little-endian (offset uint16, length uint16) ────────
_SLOT_STRUCT = struct.Struct("<HH")
assert _SLOT_STRUCT.size == SLOT_ENTRY_SIZE


# ── Page ──────────────────────────────────────────────────────────────────────

class Page:
    """
    A single 4096-byte database page.

    Supports:
      - Version 1: Sequential append format (legacy)
      - Version 2: Slotted-page architecture with dynamic slot directory and compaction

    Internally stores:
      - header : PageHeader  (mutable, re-packed on write)
      - body   : bytearray   (PAGE_BODY_SIZE bytes of payload)
    """

    __slots__ = ("header", "body")

    def __init__(self, header: PageHeader, body: bytearray) -> None:
        if len(body) != PAGE_BODY_SIZE:
            raise ValueError(
                f"Page body must be exactly {PAGE_BODY_SIZE} bytes, got {len(body)}"
            )
        self.header = header
        self.body = body

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        page_id: int,
        page_type: int = PAGE_TYPE_DATA,
        version: int = FORMAT_VERSION,
    ) -> "Page":
        """Create a blank, zeroed page with the given *page_id* and format *version*."""
        reserved = PAGE_BODY_SIZE if version >= FORMAT_VERSION_V2 else 0
        header = PageHeader(
            page_id=page_id,
            page_type=page_type,
            version=version,
            reserved=reserved,
        )
        body = bytearray(PAGE_BODY_SIZE)
        return cls(header, body)

    @classmethod
    def from_bytes(cls, data: bytes, *, validate: bool = True) -> "Page":
        """
        Deserialise a page from exactly PAGE_SIZE bytes.

        Parameters
        ----------
        data:
            Raw bytes from disk.
        validate:
            When True (default), verify magic bytes and CRC32 checksum.
            Set to False only in tests that deliberately corrupt pages.
        """
        if len(data) != PAGE_SIZE:
            raise PageCorruptedError(
                f"Expected {PAGE_SIZE} bytes, got {len(data)}"
            )

        header = PageHeader.unpack(data)

        if validate:
            # Check magic
            if header.magic != PAGE_MAGIC:
                raise PageCorruptedError(
                    f"Bad magic bytes: {header.magic!r} (expected {PAGE_MAGIC!r})"
                )
            # Check version (supports V1 and V2)
            if header.version not in (FORMAT_VERSION_V1, FORMAT_VERSION_V2):
                raise PageCorruptedError(
                    f"Unsupported format version {header.version} "
                    f"(this engine supports versions {FORMAT_VERSION_V1}, {FORMAT_VERSION_V2})"
                )
            # Verify checksum over body
            body_bytes = data[PAGE_HEADER_SIZE:]
            computed = zlib.crc32(body_bytes) & 0xFFFF_FFFF
            if computed != header.checksum:
                raise PageCorruptedError(
                    f"CRC32 mismatch on page {header.page_id}: "
                    f"stored={header.checksum:#010x}, computed={computed:#010x}"
                )

        body = bytearray(data[PAGE_HEADER_SIZE:])
        return cls(header, body)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """
        Serialise to exactly PAGE_SIZE bytes, recomputing the CRC32 checksum.
        """
        # Recompute checksum before packing
        self.header.checksum = zlib.crc32(bytes(self.body)) & 0xFFFF_FFFF
        return self.header.pack() + bytes(self.body)

    # ── body helpers ──────────────────────────────────────────────────────────

    def read_body(self, offset: int, length: int) -> bytes:
        """Read *length* bytes from the body at *offset*."""
        return bytes(self.body[offset: offset + length])

    def write_body_bytes(self, offset: int, data: bytes) -> None:
        """Write *data* into the body at *offset*."""
        end = offset + len(data)
        if end > PAGE_BODY_SIZE:
            raise ValueError(
                f"Write would exceed page body: offset={offset}, "
                f"len={len(data)}, body_size={PAGE_BODY_SIZE}"
            )
        self.body[offset:end] = data

    @property
    def free_space(self) -> int:
        """Bytes available for new records in this page."""
        if self.header.version == FORMAT_VERSION_V1:
            return PAGE_BODY_SIZE - self.header.free_offset
        return max(0, self.header.upper_offset - self.header.free_offset)

    # ── Slotted-page directory operations (Phase 8+ V2) ────────────────────────

    @property
    def is_slotted(self) -> bool:
        """True if this page uses the slotted directory layout (V2)."""
        return self.header.version >= FORMAT_VERSION_V2

    @property
    def slot_count(self) -> int:
        """Total slots currently allocated in the directory."""
        if not self.is_slotted:
            return self.header.num_records
        return self.header.free_offset // SLOT_ENTRY_SIZE

    def get_slot(self, slot_idx: int) -> tuple[int, int]:
        """Return (offset, length) for slot at *slot_idx*."""
        pos = slot_idx * SLOT_ENTRY_SIZE
        if pos + SLOT_ENTRY_SIZE > self.header.free_offset:
            raise IndexError(f"Slot index {slot_idx} out of range")
        return _SLOT_STRUCT.unpack_from(self.body, pos)

    def set_slot(self, slot_idx: int, offset: int, length: int) -> None:
        """Set (offset, length) for slot at *slot_idx*."""
        pos = slot_idx * SLOT_ENTRY_SIZE
        _SLOT_STRUCT.pack_into(self.body, pos, offset, length)

    def can_fit(self, rec_len: int) -> bool:
        """Check if *rec_len* can be inserted without compaction."""
        # Find if a tombstoned slot is reusable
        reusable = any(
            self.get_slot(i) == (0, 0)
            for i in range(self.slot_count)
        )
        slot_overhead = 0 if reusable else SLOT_ENTRY_SIZE
        return (rec_len + slot_overhead) <= self.free_space

    def can_compact(self, needed_bytes: int) -> bool:
        """Check if compacting dead/tombstoned space would yield at least *needed_bytes*."""
        active_bytes = sum(
            ln for off, ln in (self.get_slot(i) for i in range(self.slot_count))
            if off != 0 and ln != 0
        )
        max_possible = PAGE_BODY_SIZE - active_bytes - (self.slot_count * SLOT_ENTRY_SIZE)
        return max_possible >= needed_bytes

    def insert_record(self, raw: bytes) -> tuple[int, int]:
        """
        Insert a serialized record into a slotted page.

        Returns
        -------
        tuple[int, int]
            (slot_idx, payload_offset)

        Raises
        ------
        ValueError
            If record exceeds page free space even after compaction.
        """
        rec_len = len(raw)

        # 1. Look for a reusable deleted slot (offset == 0 and length == 0)
        reused_slot_idx: int | None = None
        for idx in range(self.slot_count):
            off, ln = self.get_slot(idx)
            if off == 0 and ln == 0:
                reused_slot_idx = idx
                break

        slot_overhead = 0 if reused_slot_idx is not None else SLOT_ENTRY_SIZE
        total_needed = rec_len + slot_overhead

        if total_needed > self.free_space:
            # Try defragmentation
            if self.can_compact(total_needed):
                self.compact()
                # Re-check reusable slot
                reused_slot_idx = None
                for idx in range(self.slot_count):
                    off, ln = self.get_slot(idx)
                    if off == 0 and ln == 0:
                        reused_slot_idx = idx
                        break
                slot_overhead = 0 if reused_slot_idx is not None else SLOT_ENTRY_SIZE
                total_needed = rec_len + slot_overhead

        if total_needed > self.free_space:
            raise ValueError(
                f"Record of {rec_len} bytes does not fit in page (free space: {self.free_space})"
            )

        # Allocate backward from upper_offset
        new_upper = self.header.upper_offset - rec_len
        self.write_body_bytes(new_upper, raw)
        self.header.upper_offset = new_upper

        if reused_slot_idx is not None:
            slot_idx = reused_slot_idx
            self.set_slot(slot_idx, new_upper, rec_len)
        else:
            slot_idx = self.slot_count
            self.set_slot(slot_idx, new_upper, rec_len)
            self.header.free_offset += SLOT_ENTRY_SIZE

        self.header.num_records += 1
        return slot_idx, new_upper

    def delete_slot(self, slot_idx: int) -> bool:
        """Mark record at *slot_idx* as deleted."""
        off, ln = self.get_slot(slot_idx)
        if off == 0 and ln == 0:
            return False

        # Flip tombstone flag in the record header if present
        FLAGS_OFFSET = 10
        if off + FLAGS_OFFSET < PAGE_BODY_SIZE:
            from neurochat.neurolinkdb.constants import RECORD_FLAG_DELETED
            self.body[off + FLAGS_OFFSET] |= RECORD_FLAG_DELETED

        self.set_slot(slot_idx, 0, 0)
        if self.header.num_records > 0:
            self.header.num_records -= 1
        return True

    def update_slot(self, slot_idx: int, new_raw: bytes) -> bool:
        """
        Update record at *slot_idx* with *new_raw*.

        Returns True if the record fit and was updated in this page; False otherwise.
        """
        off, ln = self.get_slot(slot_idx)
        new_len = len(new_raw)

        if new_len <= ln:
            # Overwrite in-place
            self.write_body_bytes(off, new_raw)
            self.set_slot(slot_idx, off, new_len)
            return True

        # Need new allocation: check if free space permits
        if new_len <= self.free_space:
            new_upper = self.header.upper_offset - new_len
            self.write_body_bytes(new_upper, new_raw)
            self.header.upper_offset = new_upper
            self.set_slot(slot_idx, new_upper, new_len)
            return True

        # Try compaction
        if self.can_compact(new_len):
            self.compact()
            if new_len <= self.free_space:
                new_upper = self.header.upper_offset - new_len
                self.write_body_bytes(new_upper, new_raw)
                self.header.upper_offset = new_upper
                self.set_slot(slot_idx, new_upper, new_len)
                return True

        return False

    def compact(self) -> int:
        """
        Defragment the page body: packs active records backward and trims trailing deleted slots.

        Returns the number of bytes reclaimed into contiguous free space.
        """
        old_free = self.free_space

        # 1. Collect all active records: (slot_idx, raw_bytes)
        active_records: list[tuple[int, bytes]] = []
        for idx in range(self.slot_count):
            off, ln = self.get_slot(idx)
            if off != 0 and ln != 0:
                raw = bytes(self.body[off: off + ln])
                active_records.append((idx, raw))

        # 2. Allocate fresh zeroed body
        new_body = bytearray(PAGE_BODY_SIZE)
        curr_upper = PAGE_BODY_SIZE

        # 3. Write active payloads backward from end
        for slot_idx, raw in active_records:
            rec_len = len(raw)
            curr_upper -= rec_len
            new_body[curr_upper: curr_upper + rec_len] = raw
            _SLOT_STRUCT.pack_into(new_body, slot_idx * SLOT_ENTRY_SIZE, curr_upper, rec_len)

        # 4. Trim any trailing deleted slots from directory
        last_active_slot = -1
        for idx in range(self.slot_count):
            off, ln = _SLOT_STRUCT.unpack_from(new_body, idx * SLOT_ENTRY_SIZE)
            if off != 0 and ln != 0:
                last_active_slot = idx

        new_slot_count = last_active_slot + 1
        new_free_offset = new_slot_count * SLOT_ENTRY_SIZE

        self.body = new_body
        self.header.upper_offset = curr_upper
        self.header.free_offset = new_free_offset
        self.header.num_records = len(active_records)

        new_free = self.free_space
        return max(0, new_free - old_free)

    def iter_slots(self) -> Generator[tuple[int, int, int], None, None]:
        """Yield (slot_idx, offset, length) for all active (non-deleted) slots."""
        for idx in range(self.slot_count):
            off, ln = self.get_slot(idx)
            if off != 0 and ln != 0:
                yield idx, off, ln

    def __repr__(self) -> str:
        mode = "slotted" if self.is_slotted else "v1"
        return (
            f"Page(id={self.header.page_id}, v={self.header.version}, "
            f"mode={mode}, records={self.header.num_records}, "
            f"free={self.free_space}B)"
        )

