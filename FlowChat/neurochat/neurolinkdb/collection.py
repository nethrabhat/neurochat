"""
neurolinkdb/collection.py
──────────────────────────
Collection — a named, persistent, ordered set of records stored in one or
more 4096-byte pages inside a single `.nldb` file.

File layout
───────────
  Page 0 : always the *first* data page (no separate header page in Phase 2).
  Pages 1..N : additional data pages, chained via next_page links.

Insert strategy
───────────────
  New records are appended to the last page in the chain.  When a page is
  full (no room for the new record) a fresh page is allocated and linked.

Retrieve strategy
─────────────────
  get(key)  — full linear scan (Phase 2; index comes in Phase 6).
  scan()    — yields every non-deleted Record in insertion order.

Durability
──────────
  Each insert flushes the affected page to disk immediately (no write buffer
  in Phase 2; that is deferred to Phase 7).

Limitations (Phase 2)
─────────────────────
  • No in-memory page cache.
  • No update-in-place.  Deletion is tombstone-only.
  • get(key) returns the *first* non-deleted match; duplicates are possible.
  • No overflow pages — records larger than PAGE_BODY_SIZE raise RecordTooLargeError.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, Generator, Optional

from neurochat.neurolinkdb.constants import (
    PAGE_SIZE,
    PAGE_TYPE_DATA,
    NO_NEXT_PAGE,
    SLOT_ENTRY_SIZE,
)
from neurochat.neurolinkdb.page import Page, PageCorruptedError, RecordTooLargeError
from neurochat.neurolinkdb.record import Record, serialise, deserialise
from neurochat.neurolinkdb.buffer_pool import BufferPoolManager
from neurochat.neurolinkdb.wal import WriteAheadLog


class Collection:
    """
    Persistent key-value collection backed by a chain of binary pages.

    Parameters
    ----------
    path:
        Absolute path to the `.nldb` file for this collection.
        The file is created when it does not exist.
    name:
        Human-readable collection name (for error messages).
    """

    def __init__(
        self,
        path: Path,
        name: str,
        *,
        use_wal: bool = False,
        use_buffer_pool: bool = False,
        buffer_pool_capacity: int = 16,
    ) -> None:
        self._path = path
        self.name = name
        self._next_record_id: int = 0    # monotonic counter
        self._num_pages: int = 0         # cached page count
        self._use_wal = use_wal
        self._use_buffer_pool = use_buffer_pool
        self._wal: Optional[WriteAheadLog] = None
        self._bpm: Optional[BufferPoolManager] = None

        if use_wal:
            self._wal = WriteAheadLog(path.with_suffix(".wal"))
            self._wal.recover(self._apply_recovered_page)

        if not path.exists():
            self._create_file()
        else:
            self._load_meta()

        if use_buffer_pool:
            self._bpm = BufferPoolManager(
                read_page_fn=self._raw_read_page,
                write_page_fn=self._raw_write_page,
                capacity=buffer_pool_capacity,
            )

    @property
    def buffer_pool(self) -> Optional[BufferPoolManager]:
        """Attached BufferPoolManager instance, if active."""
        return self._bpm

    @property
    def wal(self) -> Optional[WriteAheadLog]:
        """Attached WriteAheadLog instance, if active."""
        return self._wal

    # ── public API ────────────────────────────────────────────────────────────

    def insert(self, key: str, value: Any) -> Record:
        """
        Insert a new record and return the stored Record (with its assigned id).

        Raises
        ------
        RecordTooLargeError
            If the serialised record does not fit in a single page body.
        """
        record = Record(record_id=self._next_record_id, key=key, value=value)
        raw = serialise(record)           # raises RecordTooLargeError if needed

        page = self._last_page()

        if page.is_slotted:
            # Check if record fits in page
            if not page.can_fit(len(raw)):
                # Try defragmentation
                if page.can_compact(len(raw) + SLOT_ENTRY_SIZE):
                    page.compact()

            # Allocate new page if it still doesn't fit
            if not page.can_fit(len(raw)):
                new_page_id = self._num_pages
                new_page = Page.new(page_id=new_page_id, version=page.header.version)
                page.header.next_page = new_page_id
                self._write_page(page)        # flush updated chain link
                page = new_page
                self._num_pages += 1

            page.insert_record(raw)
            self._write_page(page)
            self._next_record_id += 1
            return record

        # Legacy V1 append:
        if len(raw) > page.free_space:
            new_page_id = self._num_pages
            new_page = Page.new(page_id=new_page_id, version=page.header.version)
            page.header.next_page = new_page_id
            self._write_page(page)
            page = new_page
            self._num_pages += 1

        offset = page.header.free_offset
        page.write_body_bytes(offset, raw)
        page.header.free_offset += len(raw)
        page.header.num_records += 1

        self._write_page(page)
        self._next_record_id += 1
        return record

    def update(self, key: str, new_value: Any) -> Optional[Record]:
        """
        Update the first non-deleted record matching *key* with *new_value*.

        Preserves the original record_id. In slotted pages, updates in-place if
        space permits.

        Returns
        -------
        Optional[Record]
            The updated Record if found; None if no record matching *key* exists.
        """
        page_id = 0
        while page_id != NO_NEXT_PAGE and page_id < self._num_pages:
            page = self._read_page(page_id)

            if page.is_slotted:
                for slot_idx in range(page.slot_count):
                    off, ln = page.get_slot(slot_idx)
                    if off == 0 and ln == 0:
                        continue
                    try:
                        rec, _ = deserialise(bytes(page.body), off)
                    except ValueError:
                        continue

                    if rec.key == key and not rec.deleted:
                        updated_record = Record(
                            record_id=rec.record_id,
                            key=key,
                            value=new_value,
                            deleted=False,
                        )
                        new_raw = serialise(updated_record)

                        # Try in-page update
                        if page.update_slot(slot_idx, new_raw):
                            self._write_page(page)
                            return updated_record

                        # Record expanded beyond page capacity:
                        # Delete from current page and re-insert into last page
                        page.delete_slot(slot_idx)
                        self._write_page(page)

                        # Insert into last page with the same record_id
                        last_page = self._last_page()
                        if not last_page.can_fit(len(new_raw)):
                            if last_page.can_compact(len(new_raw) + SLOT_ENTRY_SIZE):
                                last_page.compact()

                        if not last_page.can_fit(len(new_raw)):
                            new_pid = self._num_pages
                            new_pg = Page.new(page_id=new_pid, version=last_page.header.version)
                            last_page.header.next_page = new_pid
                            self._write_page(last_page)
                            last_page = new_pg
                            self._num_pages += 1

                        last_page.insert_record(new_raw)
                        self._write_page(last_page)
                        return updated_record
            else:
                # V1 legacy: mark tombstone in current page, then insert
                from neurochat.neurolinkdb.constants import RECORD_FLAG_DELETED
                offset = 0
                limit = page.header.free_offset
                while offset < limit:
                    try:
                        rec, next_offset = deserialise(bytes(page.body), offset)
                    except ValueError:
                        break

                    if rec.key == key and not rec.deleted:
                        FLAGS_OFFSET = 10
                        page.body[offset + FLAGS_OFFSET] |= RECORD_FLAG_DELETED
                        self._write_page(page)

                        updated_record = Record(
                            record_id=rec.record_id,
                            key=key,
                            value=new_value,
                            deleted=False,
                        )
                        new_raw = serialise(updated_record)

                        last_page = self._last_page()
                        if len(new_raw) > last_page.free_space:
                            new_pid = self._num_pages
                            new_pg = Page.new(page_id=new_pid, version=last_page.header.version)
                            last_page.header.next_page = new_pid
                            self._write_page(last_page)
                            last_page = new_pg
                            self._num_pages += 1

                        off = last_page.header.free_offset
                        last_page.write_body_bytes(off, new_raw)
                        last_page.header.free_offset += len(new_raw)
                        last_page.header.num_records += 1
                        self._write_page(last_page)
                        return updated_record

                    offset = next_offset

            page_id = page.header.next_page

        return None

    def vacuum(self) -> dict[str, int]:
        """
        Defragment all pages in the collection to reclaim tombstoned record space.

        Returns
        -------
        dict[str, int]
            {"pages_compacted": int, "bytes_reclaimed": int}
        """
        pages_compacted = 0
        bytes_reclaimed = 0

        page_id = 0
        while page_id != NO_NEXT_PAGE and page_id < self._num_pages:
            page = self._read_page(page_id)
            if page.is_slotted:
                reclaimed = page.compact()
                if reclaimed > 0:
                    pages_compacted += 1
                    bytes_reclaimed += reclaimed
                    self._write_page(page)
            page_id = page.header.next_page

        return {
            "pages_compacted": pages_compacted,
            "bytes_reclaimed": bytes_reclaimed,
        }

    def get(self, key: str) -> Optional[Record]:
        """
        Return the first non-deleted record with the given *key*, or None.

        Complexity: O(N) — full linear scan across all pages.
        """
        for record in self.scan():
            if record.key == key:
                return record
        return None

    def get_all(self, key: str) -> list[Record]:
        """
        Return all non-deleted records matching *key* in insertion order.
        """
        return [r for r in self.scan() if r.key == key]

    def scan(self) -> Generator[Record, None, None]:
        """
        Yield every non-deleted Record in insertion order across all pages.
        """
        page_id = 0
        while page_id != NO_NEXT_PAGE and page_id < self._num_pages:
            page = self._read_page(page_id)
            yield from _iter_page_records(page)
            page_id = page.header.next_page

    def count(self) -> int:
        """Return total number of non-deleted records (full scan)."""
        return sum(1 for _ in self.scan())

    def delete(self, key: str) -> int:
        """
        Tombstone-delete every record matching *key*.

        Returns
        -------
        int
            Number of records marked as deleted.
        """
        deleted_count = 0
        page_id = 0
        while page_id != NO_NEXT_PAGE and page_id < self._num_pages:
            page = self._read_page(page_id)
            changed = _tombstone_in_page(page, key)
            if changed:
                self._write_page(page)
                deleted_count += changed
            page_id = page.header.next_page
        return deleted_count

    @property
    def page_count(self) -> int:
        """Number of pages allocated by this collection."""
        return self._num_pages

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, pages={self._num_pages})"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _create_file(self) -> None:
        """Initialise a new collection file with a single blank page 0."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        first_page = Page.new(page_id=0)
        with self._path.open("wb") as fh:
            fh.write(first_page.to_bytes())
        self._num_pages = 1
        self._next_record_id = 0

    def _load_meta(self) -> None:
        """
        On open of an existing file:
          - count pages from file size
          - walk the page chain to find the highest record_id seen
        """
        file_size = self._path.stat().st_size
        if file_size % PAGE_SIZE != 0:
            raise PageCorruptedError(
                f"Collection '{self.name}': file size {file_size} is not a "
                f"multiple of PAGE_SIZE {PAGE_SIZE}.  The file may be truncated."
            )
        self._num_pages = file_size // PAGE_SIZE
        if self._num_pages == 0:
            raise PageCorruptedError(
                f"Collection '{self.name}': file is empty."
            )

        # Find next available record id by scanning all pages
        max_id = -1
        page_id = 0
        while page_id != NO_NEXT_PAGE and page_id < self._num_pages:
            page = self._read_page(page_id)
            for rec in _iter_page_records(page, include_deleted=True):
                if rec.record_id > max_id:
                    max_id = rec.record_id
            page_id = page.header.next_page

        self._next_record_id = max_id + 1

    def _apply_recovered_page(self, page_id: int, raw_page: bytes) -> None:
        """Apply a recovered page from the WAL during crash recovery."""
        offset = page_id * PAGE_SIZE
        mode = "r+b" if self._path.exists() else "wb"
        with self._path.open(mode) as fh:
            fh.seek(offset)
            fh.write(raw_page)
            fh.flush()
            os.fsync(fh.fileno())

    def _raw_read_page(self, page_id: int) -> Page:
        """Direct physical read of a page from disk."""
        offset = page_id * PAGE_SIZE
        with self._path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read(PAGE_SIZE)
        if len(raw) < PAGE_SIZE:
            raise PageCorruptedError(
                f"Collection '{self.name}', page {page_id}: "
                f"read only {len(raw)} bytes (expected {PAGE_SIZE})"
            )
        return Page.from_bytes(raw)

    def _read_page(self, page_id: int) -> Page:
        """Read a page, using the buffer pool if enabled."""
        if self._bpm:
            page = self._bpm.fetch_page(page_id)
            self._bpm.unpin_page(page_id)
            return page
        return self._raw_read_page(page_id)

    def _raw_write_page(self, page: Page) -> None:
        """Direct physical write of a page to disk."""
        offset = page.header.page_id * PAGE_SIZE
        raw = page.to_bytes()
        with self._path.open("r+b") as fh:
            fh.seek(offset)
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())

    def _write_page(self, page: Page) -> None:
        """Write a page, enforcing WAL protocol and buffer pool caching."""
        # 1. Enforce WAL protocol: write & fsync log record FIRST
        if self._wal:
            self._wal.log_page_write(page.header.page_id, page.to_bytes())

        # 2. Update buffer pool or write directly to disk
        if self._bpm:
            # If in buffer pool, mark frame dirty and unpin
            if not self._bpm.is_cached(page.header.page_id):
                self._raw_write_page(page)
            else:
                self._bpm.unpin_page(page.header.page_id, is_dirty=True)
                self._bpm.flush_page(page.header.page_id)
        else:
            self._raw_write_page(page)

    def flush(self) -> None:
        """Flush buffer pool dirty frames and write WAL checkpoint."""
        if self._bpm:
            self._bpm.flush_all()
        if self._wal:
            self._wal.checkpoint()

    def _last_page(self) -> Page:
        """Return the last page in the chain."""
        page_id = 0
        page = self._read_page(page_id)
        while page.header.next_page != NO_NEXT_PAGE:
            page_id = page.header.next_page
            page = self._read_page(page_id)
        return page


# ── Module-level page iteration helpers ───────────────────────────────────────

def _iter_page_records(
    page: Page,
    *,
    include_deleted: bool = False,
) -> Generator[Record, None, None]:
    """
    Iterate over all records stored in *page*.

    Parameters
    ----------
    include_deleted:
        When False (default), tombstoned records are silently skipped.
    """
    if page.is_slotted:
        body = bytes(page.body)
        for idx in range(page.slot_count):
            off, ln = page.get_slot(idx)
            if off == 0 and ln == 0:
                continue
            try:
                record, _ = deserialise(body, off)
            except ValueError:
                continue

            if not record.deleted or include_deleted:
                yield record
    else:
        body = bytes(page.body)
        offset = 0
        limit = page.header.free_offset   # only scan bytes that have been written

        while offset < limit:
            try:
                record, next_offset = deserialise(body, offset)
            except ValueError:
                break

            if not record.deleted or include_deleted:
                yield record

            offset = next_offset


def _tombstone_in_page(page: Page, key: str) -> int:
    """
    Mark every matching record in *page* as deleted in-place.

    Returns the number of records modified.
    """
    if page.is_slotted:
        count = 0
        body = bytes(page.body)
        for idx in range(page.slot_count):
            off, ln = page.get_slot(idx)
            if off == 0 and ln == 0:
                continue
            try:
                record, _ = deserialise(body, off)
            except ValueError:
                continue

            if record.key == key and not record.deleted:
                if page.delete_slot(idx):
                    count += 1
        return count

    # V1 legacy: sequential scan
    from neurochat.neurolinkdb.constants import RECORD_FLAG_DELETED
    body = page.body
    offset = 0
    limit = page.header.free_offset
    count = 0
    FLAGS_OFFSET_IN_RECORD = 10

    while offset < limit:
        try:
            record, next_offset = deserialise(bytes(body), offset)
        except ValueError:
            break

        if record.key == key and not record.deleted:
            flags_pos = offset + FLAGS_OFFSET_IN_RECORD
            body[flags_pos] = body[flags_pos] | RECORD_FLAG_DELETED
            count += 1

        offset = next_offset

    return count

