"""
neurochat.ingestion.document_store
===================================
Persist, embed, and search ingested document chunks using NeuroLinkDB.

Architecture
------------
Each ingested file produces one or more *chunks* (via TextSplitter).  Every
chunk is stored as a record in a dedicated NeuroLinkDB collection named
``documents``.  If an embedding provider and vector store are supplied the
chunk embedding is also indexed so that semantic search works out-of-the-box.

Supported file types
--------------------
* ``.txt``  — read as UTF-8 text
* ``.pdf``  — extracted with ``pdfminer.six`` when installed; otherwise a
              ``RuntimeError`` is raised with a helpful install hint.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurochat.ingestion.splitter import TextSplitter
from neurochat.neurolinkdb.database import NeuroLinkDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """Represents one chunk of an ingested document.

    Attributes
    ----------
    doc_id:
        Unique identifier for this chunk (UUID4 string).
    filename:
        Human-readable name of the source file.
    chunk_index:
        Zero-based position of this chunk within the source document.
    chunk_text:
        The raw text content of this chunk.
    source_path:
        Absolute path of the original file (empty string if ingested from raw
        text via :meth:`DocumentStore.ingest_text`).
    ingested_at:
        ISO-8601 timestamp of when the chunk was persisted.
    """

    doc_id: str
    filename: str
    chunk_index: int
    chunk_text: str
    source_path: str
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Serialisation helpers (JSON-safe dicts for NeuroLinkDB)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        """Reconstruct a :class:`DocumentRecord` from a stored dict."""
        return cls(
            doc_id=data["doc_id"],
            filename=data["filename"],
            chunk_index=int(data["chunk_index"]),
            chunk_text=data["chunk_text"],
            source_path=data.get("source_path", ""),
            ingested_at=data.get("ingested_at", ""),
        )


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------


class DocumentStore:
    """Persist and search ingested document chunks.

    Parameters
    ----------
    db_dir:
        Directory where NeuroLinkDB stores its data files.  Created if it
        does not exist.
    splitter:
        :class:`~neurochat.ingestion.splitter.TextSplitter` instance used to
        chunk raw text.  A default instance (chunk_size=512, overlap=64) is
        created when *None* is passed.
    collection_name:
        Name of the NeuroLinkDB collection that backs this store.
    """

    _COLLECTION = "documents"

    def __init__(
        self,
        db_dir: str | Path,
        *,
        splitter: Optional[TextSplitter] = None,
        collection_name: str = _COLLECTION,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._splitter = splitter or TextSplitter()
        self._collection_name = collection_name

        self._db = NeuroLinkDB(str(self._db_dir))
        self._col = self._db.collection(self._collection_name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        filename: str,
        *,
        embedder=None,
        vector_store=None,
        source_path: str = "",
    ) -> List[DocumentRecord]:
        """Chunk *text* and persist every chunk to the document collection.

        Parameters
        ----------
        text:
            Raw text content to ingest.
        filename:
            Logical name used to identify the source (shown in UI).
        embedder:
            Optional embedding provider (implements ``embed(text) -> List[float]``).
        vector_store:
            Optional :class:`~neurochat.storage.vector_store.VectorStore`.
            When both *embedder* and *vector_store* are given the chunk
            embedding is inserted so semantic search works immediately.
        source_path:
            Filesystem path of the origin file (empty for in-memory text).

        Returns
        -------
        List[DocumentRecord]
            One record per chunk produced by the splitter.
        """
        chunks = self._splitter.split(text)
        if not chunks:
            logger.warning("ingest_text: no chunks produced for '%s'", filename)
            return []

        records: List[DocumentRecord] = []
        for idx, chunk in enumerate(chunks):
            rec = DocumentRecord(
                doc_id=str(uuid.uuid4()),
                filename=filename,
                chunk_index=idx,
                chunk_text=chunk,
                source_path=source_path,
            )
            self._col.insert(rec.doc_id, rec.to_dict())

            # Optionally embed and index
            if embedder is not None and vector_store is not None:
                try:
                    embedding = embedder.embed_text(chunk)
                    vector_store.add_vector(
                        source_id=rec.doc_id,
                        vector=embedding,
                        model_name=getattr(embedder, "model_name", "mock"),
                        metadata={
                            "text": chunk,
                            "filename": filename,
                            "chunk_index": idx,
                            "source": "document",
                        },
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Failed to embed chunk %d of '%s': %s", idx, filename, exc
                    )

            records.append(rec)

        logger.info(
            "Ingested '%s': %d chunk(s) stored in collection '%s'",
            filename,
            len(records),
            self._collection_name,
        )
        return records

    def ingest_file(
        self,
        path: str | Path,
        *,
        embedder=None,
        vector_store=None,
    ) -> List[DocumentRecord]:
        """Read a file from disk and ingest its text content.

        Supported formats
        -----------------
        * ``.txt`` — UTF-8 text
        * ``.pdf`` — requires ``pdfminer.six``; raises :exc:`RuntimeError` if
                     the library is not installed.

        Parameters
        ----------
        path:
            Path to the file to ingest.

        Returns
        -------
        List[DocumentRecord]
            Chunk records produced from the file's content.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is unsupported.
        RuntimeError
            If a PDF is supplied but ``pdfminer.six`` is not installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            text = self._extract_pdf_text(path)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported: .txt, .pdf"
            )

        return self.ingest_text(
            text,
            filename=path.name,
            embedder=embedder,
            vector_store=vector_store,
            source_path=str(path.resolve()),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        *,
        embedder=None,
        vector_store=None,
        min_score: float = 0.0,
    ) -> List[Tuple[DocumentRecord, float]]:
        """Return up to *top_k* chunks most relevant to *query*.

        Strategy
        --------
        1. If *embedder* and *vector_store* are supplied, perform a semantic
           vector search and map result IDs back to :class:`DocumentRecord`.
        2. Otherwise fall back to a simple keyword (substring) scan over all
           stored chunks.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Maximum number of results.
        embedder:
            Optional embedding provider for semantic search.
        vector_store:
            Optional vector store for semantic search.
        min_score:
            Minimum similarity / score threshold (0.0 = no filter).

        Returns
        -------
        List[Tuple[DocumentRecord, float]]
            List of ``(record, score)`` pairs, sorted by descending score.
        """
        if embedder is not None and vector_store is not None:
            return self._semantic_search(
                query, top_k, embedder, vector_store, min_score
            )
        return self._keyword_search(query, top_k, min_score)

    # ------------------------------------------------------------------
    # Listing and deletion
    # ------------------------------------------------------------------

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return summary information for every unique source file ingested.

        Returns
        -------
        List[dict]
            Each dict has keys: ``filename``, ``source_path``,
            ``chunk_count``, ``ingested_at`` (earliest timestamp seen).
        """
        seen: Dict[str, Dict[str, Any]] = {}

        for record in self._col.scan():
            value = record.value
            if not isinstance(value, dict):
                continue
            fname = value.get("filename", "")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "source_path": value.get("source_path", ""),
                    "chunk_count": 0,
                    "ingested_at": value.get("ingested_at", ""),
                }
            seen[fname]["chunk_count"] += 1
            # Keep earliest timestamp
            ts = value.get("ingested_at", "")
            if ts and ts < seen[fname]["ingested_at"]:
                seen[fname]["ingested_at"] = ts

        return list(seen.values())

    def delete_document(self, filename: str) -> int:
        """Delete all chunks associated with *filename*.

        Parameters
        ----------
        filename:
            The ``filename`` field (not a filesystem path) used when the
            document was ingested.

        Returns
        -------
        int
            Number of chunk records deleted.
        """
        to_delete = [
            record.key
            for record in self._col.scan()
            if isinstance(record.value, dict) and record.value.get("filename") == filename
        ]
        for key in to_delete:
            self._col.delete(key)

        logger.info(
            "Deleted %d chunk(s) for filename '%s'", len(to_delete), filename
        )
        return len(to_delete)

    def count(self) -> int:
        """Return total number of chunk records stored."""
        return self._col.count()

    def close(self) -> None:
        """Flush and close the underlying database."""
        self._db.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Extract plain text from a PDF using pdfminer.six."""
        try:
            from pdfminer.high_level import extract_text as _extract  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires 'pdfminer.six'. "
                "Install it with:  pip install pdfminer.six"
            ) from exc
        return _extract(str(path))

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        embedder,
        vector_store,
        min_score: float,
    ) -> List[Tuple[DocumentRecord, float]]:
        """Vector-based semantic search."""
        query_vec = embedder.embed_text(query)
        raw_results = vector_store.search(query_vec, top_k=top_k)

        results: List[Tuple[DocumentRecord, float]] = []
        for res in raw_results:
            if res.score < min_score:
                continue
            # Attempt to fetch the corresponding DocumentRecord.
            # The source_id was set to the doc_id when ingesting.
            db_record = self._col.get(res.source_id)
            if db_record is None:
                # Vector was indexed but chunk record may not exist (e.g.,
                # it belongs to a conversation message, not a document chunk).
                # Skip silently.
                continue
            try:
                rec = DocumentRecord.from_dict(db_record.value)
                results.append((rec, res.score))
            except (KeyError, TypeError):
                continue

        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> List[Tuple[DocumentRecord, float]]:
        """Simple case-insensitive keyword scan (fallback when no embedder)."""
        query_lower = query.lower()
        scored: List[Tuple[DocumentRecord, float]] = []

        for record in self._col.scan():
            value = record.value
            if not isinstance(value, dict):
                continue
            text = value.get("chunk_text", "").lower()
            if not text:
                continue
            # Score = fraction of query words found in chunk
            words = query_lower.split()
            if not words:
                continue
            hits = sum(1 for w in words if w in text)
            score = hits / len(words)
            if score >= min_score:
                try:
                    rec = DocumentRecord.from_dict(value)
                    scored.append((rec, score))
                except (KeyError, TypeError):
                    continue

        # Sort by score descending, then by chunk_index ascending for stability
        scored.sort(key=lambda x: (-x[1], x[0].chunk_index))
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
