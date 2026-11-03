"""
neurolinkdb/buffer_pool.py
──────────────────────────
In-memory Buffer Pool Manager for NeuroLinkDB.

Maintains a fixed cache of page frames, tracks dirty pages, and handles
LRU (Least Recently Used) eviction with pin-count locking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from neurochat.neurolinkdb.page import Page


class BufferPoolFullError(RuntimeError):
    """Raised when all buffer pool frames are pinned and none can be evicted."""


@dataclass
class Frame:
    """An in-memory page frame within the buffer pool."""

    page: Page
    page_id: int
    pin_count: int = 1
    is_dirty: bool = False
    last_accessed: float = 0.0


class BufferPoolManager:
    """
    Manages cached memory pages between execution logic and persistent disk files.

    Parameters
    ----------
    read_page_fn:
        Function taking a page_id and returning a deserialised Page from disk.
    write_page_fn:
        Function taking a Page and persisting its bytes to disk.
    capacity:
        Maximum number of page frames allowed in memory (default: 16 = 64 KB).
    """

    def __init__(
        self,
        read_page_fn: Callable[[int], Page],
        write_page_fn: Callable[[Page], None],
        capacity: int = 16,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.read_page_fn = read_page_fn
        self.write_page_fn = write_page_fn
        self.capacity = capacity

        self.page_table: Dict[int, Frame] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.evictions: int = 0

    # ------------------------------------------------------------------
    # Buffer Pool Operations
    # ------------------------------------------------------------------

    def fetch_page(self, page_id: int) -> Page:
        """
        Retrieve a page from the buffer pool or disk.

        The retrieved page is pinned (pin_count incremented). Callers must
        call `unpin_page(page_id)` when finished.
        """
        now = time.time()

        # Cache Hit
        if page_id in self.page_table:
            frame = self.page_table[page_id]
            frame.pin_count += 1
            frame.last_accessed = now
            self.cache_hits += 1
            return frame.page

        # Cache Miss
        self.cache_misses += 1

        # Check capacity and evict if full
        if len(self.page_table) >= self.capacity:
            self._evict_lru()

        # Load page from disk
        page = self.read_page_fn(page_id)
        frame = Frame(
            page=page,
            page_id=page_id,
            pin_count=1,
            is_dirty=False,
            last_accessed=now,
        )
        self.page_table[page_id] = frame
        return page

    def unpin_page(self, page_id: int, *, is_dirty: bool = False) -> None:
        """
        Release a pin on a page, optionally marking it as modified.

        Parameters
        ----------
        page_id:
            Page to unpin.
        is_dirty:
            Set to True if the page contents were modified.
        """
        if page_id not in self.page_table:
            return

        frame = self.page_table[page_id]
        if frame.pin_count > 0:
            frame.pin_count -= 1

        if is_dirty:
            frame.is_dirty = True

    def flush_page(self, page_id: int) -> None:
        """Synchronously write page to disk if dirty and clear dirty flag."""
        if page_id not in self.page_table:
            return

        frame = self.page_table[page_id]
        if frame.is_dirty:
            self.write_page_fn(frame.page)
            frame.is_dirty = False

    def flush_all(self) -> None:
        """Synchronously flush all dirty pages to disk."""
        for frame in list(self.page_table.values()):
            if frame.is_dirty:
                self.write_page_fn(frame.page)
                frame.is_dirty = False

    def is_cached(self, page_id: int) -> bool:
        """Check if a page is currently loaded in the buffer pool."""
        return page_id in self.page_table

    def get_dirty_pages(self) -> list[int]:
        """Return list of page_ids currently marked as dirty."""
        return [f.page_id for f in self.page_table.values() if f.is_dirty]

    @property
    def hit_ratio(self) -> float:
        """Ratio of cache hits over total accesses in [0.0, 1.0]."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Eviction Internals
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Find and evict the least-recently used unpinned page."""
        candidates = [f for f in self.page_table.values() if f.pin_count == 0]
        if not candidates:
            raise BufferPoolFullError(
                f"All {self.capacity} frames in the buffer pool are currently pinned; "
                "cannot evict any page."
            )

        # Select candidate with smallest last_accessed timestamp
        lru_frame = min(candidates, key=lambda f: f.last_accessed)

        # Flush before eviction if dirty (Write-to-disk policy)
        if lru_frame.is_dirty:
            self.write_page_fn(lru_frame.page)

        del self.page_table[lru_frame.page_id]
        self.evictions += 1
