"""
neurolinkdb/__init__.py
────────────────────────
Public API for NeuroLinkDB — the custom binary storage engine.

Phase 2 exports:
  NeuroLinkDB     — database handle (open / close / collections)
  Collection      — per-collection insert / get / scan
  Record          — data record returned from the engine
  PageCorruptedError, RecordTooLargeError — exception types
"""

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.neurolinkdb.collection import Collection
from neurochat.neurolinkdb.record import Record
from neurochat.neurolinkdb.page import PageCorruptedError, RecordTooLargeError
from neurochat.neurolinkdb.buffer_pool import BufferPoolManager, BufferPoolFullError, Frame
from neurochat.neurolinkdb.wal import WriteAheadLog, LogRecord, LogRecordType

__all__ = [
    "NeuroLinkDB",
    "Collection",
    "Record",
    "PageCorruptedError",
    "RecordTooLargeError",
    "BufferPoolManager",
    "BufferPoolFullError",
    "Frame",
    "WriteAheadLog",
    "LogRecord",
    "LogRecordType",
]

