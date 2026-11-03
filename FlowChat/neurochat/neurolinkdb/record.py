"""
neurolinkdb/record.py
──────────────────────
Record serialisation / deserialisation for NeuroLinkDB.

Each record is stored as a variable-length byte sequence inside a page body:

  ┌──────────────────────────────────────────────────────────┐
  │ rec_len   uint16  — total bytes for this slot (incl hdr) │
  │ record_id uint64  — monotonic, engine-assigned           │
  │ flags     uint8   — 0x01 = deleted (tombstone)           │
  │ key_len   uint16  — byte length of the UTF-8 key         │
  │ val_len   uint32  — byte length of the JSON-encoded value│
  │ key       bytes   — key_len bytes                        │
  │ value     bytes   — val_len bytes (JSON)                 │
  └──────────────────────────────────────────────────────────┘

All integers are little-endian.
'value' is stored as UTF-8 JSON so that any JSON-serialisable Python
object can be persisted without introducing a binary-format dependency.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

from neurochat.neurolinkdb.constants import (
    RECORD_HEADER_SIZE,
    RECORD_FLAG_DELETED,
    MAX_RECORD_SIZE,
    PAGE_BODY_SIZE,
)
from neurochat.neurolinkdb.page import RecordTooLargeError


# ── Record header struct ──────────────────────────────────────────────────────
# Format:  rec_len(H2) record_id(Q8) flags(B1) key_len(H2) val_len(I4) = 17 bytes
_REC_HEADER_STRUCT = struct.Struct("<HQBHi")
assert _REC_HEADER_STRUCT.size == RECORD_HEADER_SIZE, (
    f"Record header struct size mismatch: "
    f"{_REC_HEADER_STRUCT.size} != {RECORD_HEADER_SIZE}"
)


@dataclass
class Record:
    """
    A single logical record as seen by the application layer.

    Attributes
    ----------
    record_id:
        Monotonic uint64 assigned by the engine.
    key:
        Arbitrary string key (must be valid UTF-8).
    value:
        Any JSON-serialisable Python object.
    deleted:
        True if this record carries a tombstone flag.
        Deleted records are skipped by normal scan operations.
    """

    record_id: int
    key: str
    value: Any
    deleted: bool = False

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def flags(self) -> int:
        return RECORD_FLAG_DELETED if self.deleted else 0

    def __repr__(self) -> str:
        status = " [DELETED]" if self.deleted else ""
        return f"Record(id={self.record_id}, key={self.key!r}{status})"


# ── Serialisation helpers ─────────────────────────────────────────────────────

def serialise(record: Record) -> bytes:
    """
    Encode *record* to its on-disk byte representation.

    Returns
    -------
    bytes
        Variable-length byte sequence ready to be written into a page body.

    Raises
    ------
    RecordTooLargeError
        If the encoded record exceeds MAX_RECORD_SIZE.
    ValueError
        If the key is not a valid string.
    """
    key_bytes: bytes = record.key.encode("utf-8")
    val_bytes: bytes = json.dumps(record.value, ensure_ascii=False).encode("utf-8")

    key_len = len(key_bytes)
    val_len = len(val_bytes)
    rec_len = RECORD_HEADER_SIZE + key_len + val_len

    if rec_len > MAX_RECORD_SIZE:
        raise RecordTooLargeError(
            f"Serialised record is {rec_len} bytes, "
            f"which exceeds the single-page limit of {MAX_RECORD_SIZE} bytes. "
            f"(key={record.key!r}, value_len={val_len})"
        )

    header = _REC_HEADER_STRUCT.pack(
        rec_len,
        record.record_id,
        record.flags,
        key_len,
        val_len,
    )
    return header + key_bytes + val_bytes


def deserialise(data: bytes, offset: int = 0) -> tuple[Record, int]:
    """
    Decode one record from *data* starting at *offset*.

    Returns
    -------
    (record, next_offset)
        *record*       — the decoded Record object.
        *next_offset*  — the offset immediately after this record's bytes.

    Raises
    ------
    ValueError
        If there is insufficient data or the header is malformed.
    """
    available = len(data) - offset
    if available < RECORD_HEADER_SIZE:
        raise ValueError(
            f"Not enough bytes to read record header at offset {offset}: "
            f"need {RECORD_HEADER_SIZE}, have {available}"
        )

    rec_len, record_id, flags, key_len, val_len = _REC_HEADER_STRUCT.unpack_from(
        data, offset
    )

    # Basic sanity
    if rec_len < RECORD_HEADER_SIZE:
        raise ValueError(
            f"Record at offset {offset} has invalid rec_len={rec_len} "
            f"(minimum is {RECORD_HEADER_SIZE})"
        )
    total_needed = RECORD_HEADER_SIZE + key_len + val_len
    if rec_len < total_needed:
        raise ValueError(
            f"Record at offset {offset}: rec_len={rec_len} is smaller than "
            f"header+key+val={total_needed}"
        )
    if offset + rec_len > len(data):
        raise ValueError(
            f"Record at offset {offset} extends beyond data buffer "
            f"(rec_len={rec_len}, buffer_size={len(data)})"
        )

    key_start = offset + RECORD_HEADER_SIZE
    val_start = key_start + key_len
    val_end   = val_start + val_len

    key   = data[key_start:val_start].decode("utf-8")
    value = json.loads(data[val_start:val_end].decode("utf-8"))

    deleted = bool(flags & RECORD_FLAG_DELETED)
    record  = Record(record_id=record_id, key=key, value=value, deleted=deleted)

    return record, offset + rec_len

