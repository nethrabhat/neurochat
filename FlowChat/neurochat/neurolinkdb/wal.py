"""
neurolinkdb/wal.py
──────────────────
Write-Ahead Log (WAL) and Redo Recovery for NeuroLinkDB.

Enforces the core WAL invariant: modifications must be written and fsynced
to the persistent log before dirty pages can be flushed to the database file.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

WAL_MAGIC: bytes = b"NLWL"  # 4-byte WAL file header
WAL_HEADER_SIZE: int = 21   # lsn(8) + type(1) + page_id(4) + len(4) + crc(4)

# Binary log record header: <Q B I I I (little-endian, packed)
_LOG_HEADER_STRUCT = struct.Struct("<QBIII")
assert _LOG_HEADER_STRUCT.size == WAL_HEADER_SIZE


class LogRecordType(IntEnum):
    PAGE_WRITE = 1
    PAGE_ALLOC = 2
    CHECKPOINT = 3


@dataclass
class LogRecord:
    """Parsed representation of an on-disk WAL record."""

    lsn: int
    record_type: LogRecordType
    page_id: int
    payload: bytes
    checksum: int


class WriteAheadLog:
    """
    Append-only Write-Ahead Log manager.

    Parameters
    ----------
    path:
        Absolute path to the `.wal` file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._next_lsn: int = 1
        self._fh: Optional[object] = None

        self._init_log()

    # ------------------------------------------------------------------
    # WAL Operations
    # ------------------------------------------------------------------

    def append(
        self,
        record_type: LogRecordType,
        page_id: int,
        payload: bytes,
    ) -> int:
        """
        Append a new log record to the WAL buffer.

        Returns
        -------
        int
            Assigned monotonic Log Sequence Number (LSN).
        """
        lsn = self._next_lsn
        payload_len = len(payload)
        checksum = zlib.crc32(payload) & 0xFFFF_FFFF

        header_bytes = _LOG_HEADER_STRUCT.pack(
            lsn,
            int(record_type),
            page_id,
            payload_len,
            checksum,
        )

        with self.path.open("ab") as fh:
            fh.write(header_bytes)
            if payload:
                fh.write(payload)

        self._next_lsn += 1
        return lsn

    def flush(self) -> None:
        """Synchronously force all log records to physical storage via fsync."""
        with self.path.open("a+b") as fh:
            fh.flush()
            os.fsync(fh.fileno())

    def log_page_write(self, page_id: int, page_bytes: bytes) -> int:
        """Convenience method to log a page mutation and fsync."""
        lsn = self.append(LogRecordType.PAGE_WRITE, page_id, page_bytes)
        self.flush()
        return lsn

    def checkpoint(self) -> int:
        """Write a checkpoint marker and fsync."""
        lsn = self.append(LogRecordType.CHECKPOINT, 0, b"")
        self.flush()
        return lsn

    def truncate(self) -> None:
        """Reset the WAL file after all dirty pages are safely checkpointed."""
        with self.path.open("wb") as fh:
            fh.write(WAL_MAGIC)
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------
    # Crash Recovery (REDO)
    # ------------------------------------------------------------------

    def recover(
        self,
        apply_page_fn: Callable[[int, bytes], None],
    ) -> int:
        """
        Scan log from beginning, verify checksums, and replay uncorrupted records.

        Handles interrupted/torn writes safely by halting at the first truncated
        or checksum-mismatched record.

        Parameters
        ----------
        apply_page_fn:
            Callback taking (page_id, raw_page_bytes) to overwrite database pages.

        Returns
        -------
        int
            Number of log records successfully replayed.
        """
        if not self.path.exists() or self.path.stat().st_size <= len(WAL_MAGIC):
            return 0

        replayed_count = 0
        with self.path.open("rb") as fh:
            magic = fh.read(len(WAL_MAGIC))
            if magic != WAL_MAGIC:
                return 0

            while True:
                header_data = fh.read(WAL_HEADER_SIZE)
                if len(header_data) < WAL_HEADER_SIZE:
                    # Graceful or interrupted EOF at record boundary
                    break

                (
                    lsn,
                    rec_type_val,
                    page_id,
                    payload_len,
                    stored_checksum,
                ) = _LOG_HEADER_STRUCT.unpack(header_data)

                payload = fh.read(payload_len)
                if len(payload) < payload_len:
                    # Torn write / partial payload: stop safely
                    break

                # Verify payload CRC32 checksum
                computed_crc = zlib.crc32(payload) & 0xFFFF_FFFF
                if computed_crc != stored_checksum:
                    # Corrupted payload: stop recovery at last clean point
                    break

                # Replay valid write actions
                rec_type = LogRecordType(rec_type_val)
                if rec_type in (LogRecordType.PAGE_WRITE, LogRecordType.PAGE_ALLOC):
                    apply_page_fn(page_id, payload)
                    replayed_count += 1

                self._next_lsn = max(self._next_lsn, lsn + 1)

        return replayed_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        """Ensure WAL file exists and contains magic identifier."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as fh:
                fh.write(WAL_MAGIC)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            # Find highest LSN already in log
            self._scan_highest_lsn()

    def _scan_highest_lsn(self) -> None:
        """Scan log file to set next_lsn appropriately."""
        if self.path.stat().st_size <= len(WAL_MAGIC):
            return

        with self.path.open("rb") as fh:
            fh.seek(len(WAL_MAGIC))
            while True:
                hdr = fh.read(WAL_HEADER_SIZE)
                if len(hdr) < WAL_HEADER_SIZE:
                    break
                lsn, _, _, payload_len, _ = _LOG_HEADER_STRUCT.unpack(hdr)
                fh.seek(payload_len, os.SEEK_CUR)
                self._next_lsn = max(self._next_lsn, lsn + 1)
