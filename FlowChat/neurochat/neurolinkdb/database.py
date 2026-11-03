"""
neurolinkdb/database.py
────────────────────────
NeuroLinkDB — top-level database handle.

A database is simply a directory on disk.  Each collection is stored as a
separate `<collection_name>.nldb` file inside that directory.

Usage::

    db = NeuroLinkDB("path/to/mydb")
    col = db.collection("messages")
    col.insert("msg:1", {"role": "user", "content": "Hello"})
    record = col.get("msg:1")
    db.close()

    # Context manager form
    with NeuroLinkDB("path/to/mydb") as db:
        col = db.collection("messages")
        ...

Phase 2 limitations
────────────────────
  • No cross-collection transactions.
  • Collection objects are not cached across calls to `collection()` — each
    call reopens the file.  Phase 7 will add a proper buffer pool.
  • Concurrent access from multiple processes is not safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from neurochat.neurolinkdb.collection import Collection


_COLLECTION_SUFFIX = ".nldb"


class NeuroLinkDB:
    """
    Database handle for NeuroLinkDB.

    Parameters
    ----------
    directory:
        Path to the database directory.  Created automatically if it does not
        exist.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._open_collections: list[Collection] = []

    # ── collection management ─────────────────────────────────────────────────

    def collection(
        self,
        name: str,
        *,
        use_wal: bool = False,
        use_buffer_pool: bool = False,
        buffer_pool_capacity: int = 16,
    ) -> Collection:
        """
        Open (or create) the collection named *name*.

        Collection names must be non-empty strings containing only ASCII
        letters, digits, underscores, and hyphens.

        Parameters
        ----------
        name:
            Collection name.
        use_wal:
            Enable Write-Ahead Logging for crash recovery (Phase 7).
        use_buffer_pool:
            Enable in-memory LRU buffer pool caching (Phase 7).
        buffer_pool_capacity:
            Number of cached page frames in the buffer pool.

        Returns
        -------
        Collection
            A Collection object backed by ``<db_dir>/<name>.nldb``.

        Raises
        ------
        ValueError
            If the database has been closed or *name* is invalid.
        """
        self._assert_open()
        _validate_name(name)
        path = self._dir / f"{name}{_COLLECTION_SUFFIX}"
        col = Collection(
            path=path,
            name=name,
            use_wal=use_wal,
            use_buffer_pool=use_buffer_pool,
            buffer_pool_capacity=buffer_pool_capacity,
        )
        self._open_collections.append(col)
        return col

    def collection_names(self) -> list[str]:
        """Return the names of all existing collections in this database."""
        self._assert_open()
        return [
            p.stem
            for p in self._dir.iterdir()
            if p.is_file() and p.suffix == _COLLECTION_SUFFIX
        ]

    def drop_collection(self, name: str) -> bool:
        """
        Permanently delete a collection file and any associated WAL file.

        Returns
        -------
        bool
            True if the collection existed and was deleted; False otherwise.
        """
        self._assert_open()
        _validate_name(name)
        path = self._dir / f"{name}{_COLLECTION_SUFFIX}"
        wal_path = self._dir / f"{name}.wal"
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if wal_path.exists():
            wal_path.unlink()
        return deleted

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Mark this database handle as closed and flush active collections.
        """
        for col in self._open_collections:
            try:
                col.flush()
            except Exception:
                pass
        self._open_collections.clear()
        self._closed = True

    @property
    def is_open(self) -> bool:
        return not self._closed

    @property
    def directory(self) -> Path:
        return self._dir

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "NeuroLinkDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._closed:
            raise ValueError(
                f"NeuroLinkDB at '{self._dir}' has been closed and cannot be used."
            )

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"NeuroLinkDB(dir={self._dir!r}, state={state})"


# ── Validation ────────────────────────────────────────────────────────────────

import re
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("Collection name must not be empty.")
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"Invalid collection name {name!r}.  "
            "Only ASCII letters, digits, underscores, and hyphens are allowed."
        )

