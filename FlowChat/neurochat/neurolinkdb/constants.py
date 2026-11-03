"""
neurolinkdb/constants.py
────────────────────────
Shared constants for the NeuroLinkDB binary page format.

All sizes are in bytes unless stated otherwise.
"""

from __future__ import annotations

# ── Page geometry ────────────────────────────────────────────────────────────
PAGE_SIZE: int = 4096          # bytes per page
PAGE_HEADER_SIZE: int = 32     # bytes reserved for the page header
PAGE_BODY_SIZE: int = PAGE_SIZE - PAGE_HEADER_SIZE  # usable payload area

# ── Magic bytes & format version ─────────────────────────────────────────────
PAGE_MAGIC: bytes = b"NLDB"    # 4-byte file / page identifier
FORMAT_VERSION_V1: int = 1     # legacy sequential-append page format
FORMAT_VERSION_V2: int = 2     # slotted-page format with slot directory & compaction
FORMAT_VERSION: int = 2        # default format version for new pages

# ── Slot entry geometry (Phase 8+) ───────────────────────────────────────────
SLOT_ENTRY_SIZE: int = 4       # uint16 offset + uint16 length in slot directory

# ── Page types ────────────────────────────────────────────────────────────────
PAGE_TYPE_FREE: int = 0        # unallocated / reusable
PAGE_TYPE_DATA: int = 1        # holds serialised records
PAGE_TYPE_OVERFLOW: int = 2    # continuation of a large record (Phase 7+)

# ── Sentinel values ───────────────────────────────────────────────────────────
NO_NEXT_PAGE: int = 0xFFFF_FFFF   # page_id meaning "no next page"

# ── Record flags ──────────────────────────────────────────────────────────────
RECORD_FLAG_DELETED: int = 0x01   # tombstone; skip during scan

# ── Record header fixed overhead (bytes before key/value data) ────────────────
#   rec_len  (2) + record_id (8) + flags (1) + key_len (2) + val_len (4) = 17
RECORD_HEADER_SIZE: int = 17

# ── Safety limits ─────────────────────────────────────────────────────────────
# A single record (header + key + value) must fit in one page body.
# Overflow pages are deferred to a later phase.
MAX_RECORD_SIZE: int = PAGE_BODY_SIZE   # 4064 bytes

