"""
neurolinkdb/index/ivfflat.py
────────────────────────────
Custom IVFFlat (Inverted File Flat) index implementation for NeuroLinkDB.

Partitions vector space into Voronoi cells using spherical K-Means.
Queries probe only the nearest nprobe cells, reducing search comparisons
from O(N) to O(K + (nprobe / K) * N).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from neurochat.neurolinkdb.collection import Collection
from neurochat.neurolinkdb.index.kmeans import KMeans
from neurochat.embeddings.similarity import (
    cosine_similarity,
    normalize,
    validate_equal_dimensions,
)
from neurochat.storage.vector_store import (
    VectorRecord,
    VectorSearchResult,
    IncompatibleVectorError,
)


class IVFFlatIndex:
    """
    Inverted File Flat (IVFFlat) vector index.

    Parameters
    ----------
    dimension:
        Expected vector dimensionality.
    nlist:
        Number of Voronoi partitions (clusters).
    """

    def __init__(self, dimension: int, nlist: int = 4) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if nlist <= 0:
            raise ValueError(f"nlist must be positive, got {nlist}")

        self.dimension = dimension
        self.nlist = nlist
        self.centroids: List[List[float]] = []
        self.inverted_lists: Dict[int, List[VectorRecord]] = {
            i: [] for i in range(nlist)
        }
        self.is_trained: bool = False
        self.trained_at: float = 0.0
        self.total_vectors: int = 0

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def build(self, records: Sequence[VectorRecord]) -> IVFFlatIndex:
        """
        Train centroids on input records and populate inverted file lists.

        Parameters
        ----------
        records:
            Vector records to index.

        Returns
        -------
        IVFFlatIndex
            Fitted index instance.
        """
        if not records:
            raise ValueError("Cannot build IVFFlat index on empty record set.")

        for r in records:
            if r.dimension != self.dimension:
                raise IncompatibleVectorError(
                    f"Record {r.vector_id} dimension {r.dimension} does not "
                    f"match index dimension {self.dimension}."
                )

        vectors = [r.vector for r in records]
        k = min(self.nlist, len(records))

        # Train KMeans clustering
        kmeans = KMeans(n_clusters=k, max_iter=30, random_seed=42)
        kmeans.fit(vectors)

        self.centroids = kmeans.centroids
        self.nlist = len(self.centroids)
        self.inverted_lists = {i: [] for i in range(self.nlist)}

        # Assign records to their closest centroid
        assignments = kmeans.predict(vectors)
        for record, cluster_idx in zip(records, assignments):
            self.inverted_lists[cluster_idx].append(record)

        self.is_trained = True
        self.trained_at = time.time()
        self.total_vectors = len(records)
        return self

    # ------------------------------------------------------------------
    # Approximate Nearest-Neighbor Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        nprobe: int = 1,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """
        Query the index probing only the top `nprobe` nearest centroid cells.

        Parameters
        ----------
        query_vector:
            Query embedding vector.
        top_k:
            Number of top nearest neighbors to return.
        nprobe:
            Number of closest centroids to probe (1 <= nprobe <= nlist).
        threshold:
            Minimum similarity threshold.
        filter_metadata:
            Optional metadata filters that records must satisfy.

        Returns
        -------
        List[VectorSearchResult]
            Ranked list of top-k matches.
        """
        if not self.is_trained or not self.centroids:
            return []

        q_norm = normalize(query_vector)
        if len(q_norm) != self.dimension:
            raise IncompatibleVectorError(
                f"Query dimension {len(q_norm)} != index dimension {self.dimension}"
            )

        probes = max(1, min(nprobe, self.nlist))

        # Step 1: Find closest centroids
        centroid_scores: List[tuple[int, float]] = []
        for idx, c in enumerate(self.centroids):
            sim = cosine_similarity(q_norm, c)
            centroid_scores.append((idx, sim))

        centroid_scores.sort(key=lambda item: item[1], reverse=True)
        probed_clusters = [idx for idx, _ in centroid_scores[:probes]]

        # Step 2: Scan only records in the probed inverted lists
        candidates: List[VectorSearchResult] = []
        for c_idx in probed_clusters:
            for rec in self.inverted_lists.get(c_idx, []):
                # Apply metadata filter
                if filter_metadata:
                    m = rec.metadata
                    match = True
                    for k, v in filter_metadata.items():
                        if m.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                sim = cosine_similarity(q_norm, rec.vector)
                if threshold is not None and sim < threshold:
                    continue

                candidates.append(
                    VectorSearchResult(
                        vector_id=rec.vector_id,
                        source_id=rec.source_id,
                        score=sim,
                        metadata=rec.metadata,
                        vector=rec.vector,
                    )
                )

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, collection: Collection) -> None:
        """Persist index metadata, centroids, and inverted lists into NeuroLinkDB."""
        # 1. Save metadata
        meta = {
            "dimension": self.dimension,
            "nlist": self.nlist,
            "is_trained": self.is_trained,
            "trained_at": self.trained_at,
            "total_vectors": self.total_vectors,
            "centroids": self.centroids,
        }
        collection.insert("__ivf_meta__", meta)

        # 2. Save inverted list items individually
        for c_idx, records in self.inverted_lists.items():
            for rec in records:
                key = f"inv:{c_idx}:{rec.vector_id}"
                collection.insert(key, rec.to_dict())

    @classmethod
    def load(cls, collection: Collection) -> Optional[IVFFlatIndex]:
        """Load index from an existing NeuroLinkDB collection."""
        meta_rec = collection.get("__ivf_meta__")
        if not meta_rec or not isinstance(meta_rec.value, dict):
            return None

        val = meta_rec.value
        index = cls(dimension=val["dimension"], nlist=val["nlist"])
        index.centroids = [list(c) for c in val.get("centroids", [])]
        index.is_trained = bool(val.get("is_trained", False))
        index.trained_at = float(val.get("trained_at", 0.0))
        index.total_vectors = int(val.get("total_vectors", 0))

        # Reconstruct inverted lists
        index.inverted_lists = {i: [] for i in range(index.nlist)}
        for record in collection.scan():
            if record.key.startswith("inv:") and isinstance(record.value, dict):
                parts = record.key.split(":")
                if len(parts) >= 3:
                    try:
                        c_idx = int(parts[1])
                        v_rec = VectorRecord.from_dict(record.value)
                        index.inverted_lists.setdefault(c_idx, []).append(v_rec)
                    except (ValueError, KeyError):
                        continue

        return index

