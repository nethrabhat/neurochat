"""
storage/vector_store.py
───────────────────────
Vector storage and exact k-NN similarity search backed by NeuroLinkDB.

Manages persistent vector collections, enforces dimensionality consistency,
and executes exact nearest-neighbor ranking using cosine similarity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.neurolinkdb.collection import Collection
from neurochat.embeddings.similarity import cosine_similarity

if TYPE_CHECKING:
    from neurochat.neurolinkdb.index.ivfflat import IVFFlatIndex


@dataclass
class VectorRecord:
    """
    Persistent schema for an embedding vector record.

    Attributes
    ----------
    vector_id:
        Unique UUID identifying this vector record.
    source_id:
        Identifier of the source entity (e.g. message_id or chunk_id).
    vector:
        Float components of the embedding vector.
    model_name:
        Identifier of the embedding model that generated this vector.
    dimension:
        Dimensionality of the vector.
    metadata:
        Arbitrary metadata dictionary for relational filtering (e.g. conversation_id).
    """

    vector_id: str
    source_id: str
    vector: List[float]
    model_name: str
    dimension: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        """Reconstruct a VectorRecord from a dictionary."""
        return cls(
            vector_id=str(data["vector_id"]),
            source_id=str(data["source_id"]),
            vector=[float(x) for x in data["vector"]],
            model_name=str(data["model_name"]),
            dimension=int(data["dimension"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VectorSearchResult:
    """
    Search result item returned by nearest-neighbor queries.

    Attributes
    ----------
    vector_id:
        ID of the matching vector record.
    source_id:
        Associated entity ID (e.g. message_id).
    score:
        Cosine similarity score in [-1.0, 1.0].
    metadata:
        Associated metadata dictionary.
    vector:
        The matched vector.
    """

    vector_id: str
    source_id: str
    score: float
    metadata: Dict[str, Any]
    vector: List[float]


class IncompatibleVectorError(ValueError):
    """Raised when a vector violates dimension or model constraints."""


class VectorStore:
    """
    Exact vector store maintaining embedding vectors within a NeuroLinkDB collection.

    Parameters
    ----------
    db:
        Active NeuroLinkDB database instance.
    collection_name:
        Collection name for storing vectors (default: 'vectors').
    dimension:
        Expected vector dimension. If specified, any vector with differing length
        is rejected. If omitted, the dimension is locked to the first inserted vector.
    model_name:
        Expected embedding model identity. If specified, vectors from differing
        models are rejected.
    """

    DEFAULT_COLLECTION_NAME: str = "vectors"
    _META_KEY: str = "__vector_store_metadata__"

    def __init__(
        self,
        db: NeuroLinkDB,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        dimension: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._collection: Collection = db.collection(collection_name)
        self._dimension = dimension
        self._model_name = model_name
        self._index_collection_name = f"{collection_name}_ivfflat"
        self._index: Optional[IVFFlatIndex] = None

        self._load_or_persist_meta()
        self._try_load_index()

    # ------------------------------------------------------------------
    # Public Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> Optional[int]:
        """Expected dimensionality of this vector collection."""
        return self._dimension

    @property
    def model_name(self) -> Optional[str]:
        """Expected model identity of this vector collection."""
        return self._model_name

    @property
    def has_index(self) -> bool:
        """True if an IVFFlat index is active and trained."""
        return bool(self._index and self._index.is_trained)

    @property
    def index(self) -> Optional[IVFFlatIndex]:
        """The underlying IVFFlat index, if built."""
        return self._index

    # ------------------------------------------------------------------
    # Vector Operations
    # ------------------------------------------------------------------

    def add_vector(
        self,
        source_id: str,
        vector: Sequence[float],
        model_name: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> VectorRecord:
        """
        Store a vector associated with a source entity.

        Parameters
        ----------
        source_id:
            ID of the source entity (e.g. message_id).
        vector:
            Numerical vector components.
        model_name:
            Identifier of the embedding model.
        metadata:
            Optional metadata dictionary for filtering.
        vector_id:
            Optional explicit UUID; generated automatically if omitted.

        Returns
        -------
        VectorRecord
            The stored vector record.

        Raises
        ------
        IncompatibleVectorError
            If vector dimension or model_name contradicts collection constraints.
        """
        vec_dim = len(vector)
        if vec_dim == 0:
            raise IncompatibleVectorError("Vector cannot be empty.")

        # Validate or lock dimensionality
        if self._dimension is None:
            self._dimension = vec_dim
            self._model_name = model_name
            self._save_meta()
        else:
            if vec_dim != self._dimension:
                raise IncompatibleVectorError(
                    f"Vector dimension {vec_dim} does not match collection "
                    f"dimension {self._dimension}."
                )
            if self._model_name and model_name != self._model_name:
                raise IncompatibleVectorError(
                    f"Model name '{model_name}' does not match collection "
                    f"model '{self._model_name}'."
                )

        vid = vector_id or str(uuid.uuid4())
        record = VectorRecord(
            vector_id=vid,
            source_id=source_id,
            vector=list(vector),
            model_name=model_name,
            dimension=vec_dim,
            metadata=dict(metadata or {}),
        )

        key = f"vec:{vid}"
        self._collection.insert(key, record.to_dict())
        return record

    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Retrieve a vector record by vector_id."""
        rec = self._collection.get(f"vec:{vector_id}")
        if rec and isinstance(rec.value, dict):
            return VectorRecord.from_dict(rec.value)
        return None

    def get_by_source_id(self, source_id: str) -> List[VectorRecord]:
        """Find all vectors linked to a particular source_id."""
        results: List[VectorRecord] = []
        for rec in self._collection.scan():
            if rec.key.startswith("vec:") and isinstance(rec.value, dict):
                if rec.value.get("source_id") == source_id:
                    results.append(VectorRecord.from_dict(rec.value))
        return results

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all vectors associated with source_id.

        Returns
        -------
        int
            Number of deleted vector records.
        """
        keys_to_delete: List[str] = []
        for rec in self._collection.scan():
            if rec.key.startswith("vec:") and isinstance(rec.value, dict):
                if rec.value.get("source_id") == source_id:
                    keys_to_delete.append(rec.key)

        deleted = 0
        for k in keys_to_delete:
            deleted += self._collection.delete(k)
        return deleted

    def count(self) -> int:
        """Return total number of active vectors in the store."""
        count = 0
        for rec in self._collection.scan():
            if rec.key.startswith("vec:"):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Index Management (Phase 6 IVFFlat)
    # ------------------------------------------------------------------

    def build_index(self, nlist: int = 4) -> IVFFlatIndex:
        """
        Train and persist an IVFFlat index from all stored vectors.

        Parameters
        ----------
        nlist:
            Number of Voronoi partitions (clusters).

        Returns
        -------
        IVFFlatIndex
            The newly built index.
        """
        if self._dimension is None:
            raise ValueError("Cannot build index on empty vector store.")

        from neurochat.neurolinkdb.index.ivfflat import IVFFlatIndex

        records = self.get_all_records()
        if not records:
            raise ValueError("Cannot build index: no vectors present in store.")

        index = IVFFlatIndex(dimension=self._dimension, nlist=nlist)
        index.build(records)

        # Persist index into its own collection
        index_col = self._db.collection(self._index_collection_name)
        # Clear previous index records if any
        self._db.drop_collection(self._index_collection_name)
        index_col = self._db.collection(self._index_collection_name)
        index.save(index_col)

        self._index = index
        return index

    def rebuild_index(self, nlist: Optional[int] = None) -> IVFFlatIndex:
        """Rebuild the index to capture recent inserts or deletions."""
        current_nlist = nlist or (self._index.nlist if self._index else 4)
        return self.build_index(nlist=current_nlist)

    def get_all_records(self) -> List[VectorRecord]:
        """Return all active VectorRecord items stored in the collection."""
        records: List[VectorRecord] = []
        for rec in self._collection.scan():
            if rec.key.startswith("vec:") and isinstance(rec.value, dict):
                try:
                    records.append(VectorRecord.from_dict(rec.value))
                except (KeyError, ValueError, TypeError):
                    continue
        return records

    # ------------------------------------------------------------------
    # Hybrid Search (Exact & Approximate)
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
        use_index: bool = False,
        nprobe: int = 1,
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid vector search with metadata filtering and optional IVFFlat index.

        Parameters
        ----------
        query_vector:
            Query embedding vector.
        top_k:
            Maximum number of top-ranked results to return.
        threshold:
            Minimum similarity threshold (inclusive).
        filter_metadata:
            Optional key-value pairs that candidate record metadata must match.
        filter_fn:
            Optional arbitrary predicate evaluated against record metadata.
        min_timestamp:
            Optional minimum UNIX timestamp bound for metadata['timestamp'].
        max_timestamp:
            Optional maximum UNIX timestamp bound for metadata['timestamp'].
        use_index:
            When True and an IVFFlat index is built, executes approximate search.
            When False (default), executes exhaustive exact search.
        nprobe:
            Number of closest centroids to probe if use_index is True.

        Returns
        -------
        List[VectorSearchResult]
            Top-k matches sorted descending by similarity score.
        """
        if top_k <= 0:
            return []

        q_dim = len(query_vector)
        if self._dimension is not None and q_dim != self._dimension:
            raise IncompatibleVectorError(
                f"Query vector dimension {q_dim} does not match collection "
                f"dimension {self._dimension}."
            )

        def matches_hybrid_filters(metadata: Dict[str, Any]) -> bool:
            if filter_metadata:
                for k, v in filter_metadata.items():
                    if metadata.get(k) != v:
                        return False

            if filter_fn and not filter_fn(metadata):
                return False

            if min_timestamp is not None:
                ts = metadata.get("timestamp")
                if ts is None or ts < min_timestamp:
                    return False

            if max_timestamp is not None:
                ts = metadata.get("timestamp")
                if ts is None or ts > max_timestamp:
                    return False

            return True

        # Branch 1: Approximate search via IVFFlat index
        if use_index and self._index and self._index.is_trained:
            # Over-fetch from index to allow subsequent relational filter evaluation
            raw_results = self._index.search(
                query_vector,
                top_k=top_k * 3,
                nprobe=nprobe,
                threshold=threshold,
                filter_metadata=filter_metadata,
            )
            filtered = [
                res for res in raw_results if matches_hybrid_filters(res.metadata)
            ]
            filtered.sort(key=lambda r: r.score, reverse=True)
            return filtered[:top_k]

        # Branch 2: Exhaustive exact scan (baseline)
        candidates: List[VectorSearchResult] = []

        for rec in self._collection.scan():
            if not rec.key.startswith("vec:") or not isinstance(rec.value, dict):
                continue

            try:
                v_record = VectorRecord.from_dict(rec.value)
            except (KeyError, ValueError, TypeError):
                continue

            # Hybrid metadata filtering
            if not matches_hybrid_filters(v_record.metadata):
                continue

            # Compute exact cosine similarity
            score = cosine_similarity(query_vector, v_record.vector)

            # Threshold filtering
            if threshold is not None and score < threshold:
                continue

            candidates.append(
                VectorSearchResult(
                    vector_id=v_record.vector_id,
                    source_id=v_record.source_id,
                    score=score,
                    metadata=v_record.metadata,
                    vector=v_record.vector,
                )
            )

        # Sort descending by score
        candidates.sort(key=lambda res: res.score, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Internal Metadata Handling
    # ------------------------------------------------------------------

    def _load_or_persist_meta(self) -> None:
        """Load stored metadata record if present, or persist configuration."""
        meta_rec = self._collection.get(self._META_KEY)
        if meta_rec and isinstance(meta_rec.value, dict):
            stored_dim = meta_rec.value.get("dimension")
            stored_model = meta_rec.value.get("model_name")

            if self._dimension is not None and stored_dim != self._dimension:
                raise IncompatibleVectorError(
                    f"Configured dimension {self._dimension} conflicts with "
                    f"existing database dimension {stored_dim}."
                )
            if self._model_name and stored_model != self._model_name:
                raise IncompatibleVectorError(
                    f"Configured model '{self._model_name}' conflicts with "
                    f"existing database model '{stored_model}'."
                )

            self._dimension = stored_dim
            self._model_name = stored_model
        elif self._dimension is not None:
            self._save_meta()

    def _save_meta(self) -> None:
        """Save collection metadata record."""
        meta_data = {
            "dimension": self._dimension,
            "model_name": self._model_name,
        }
        self._collection.insert(self._META_KEY, meta_data)

    def _try_load_index(self) -> None:
        """Attempt to restore IVFFlat index if saved in database."""
        try:
            from neurochat.neurolinkdb.index.ivfflat import IVFFlatIndex

            index_path_col = self._db.collection(self._index_collection_name)
            loaded = IVFFlatIndex.load(index_path_col)
            if loaded and loaded.is_trained:
                self._index = loaded
        except Exception:
            self._index = None

