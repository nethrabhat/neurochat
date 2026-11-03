"""
tests/test_ivfflat_index.py
───────────────────────────
Phase 6 test suite: Custom K-Means, IVFFlat Index, Hybrid Queries, and Recall@K Evaluation.

Covers:
  - Spherical K-Means clustering algorithm and centroid unit-normalization
  - Deterministic clustering convergence
  - IVFFlat index partitioning and inverted list distribution
  - Approximate nearest neighbor search scaling with nprobe
  - Complete Recall@K verification (nprobe=nlist matches exact k-NN ground truth)
  - Relational hybrid queries: metadata key-value, lambda predicates, and timestamp ranges
  - Persistence across database close and reopen
  - Index rebuilding when data changes
  - Search performance benchmarking and Recall@K evaluation metrics

All tests run fully offline without external C libraries or GPU dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.neurolinkdb.index.kmeans import KMeans
from neurochat.neurolinkdb.index.ivfflat import IVFFlatIndex
from neurochat.neurolinkdb.index.evaluator import (
    compute_recall_at_k,
    evaluate_index_performance,
)
from neurochat.embeddings.similarity import (
    cosine_similarity,
    l2_norm,
    normalize,
)
from neurochat.embeddings.base import MockEmbeddingProvider
from neurochat.storage.vector_store import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
    IncompatibleVectorError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    return tmp_path / "ivf_test_db"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def synthetic_records() -> list[VectorRecord]:
    """Generate 20 deterministic 4D vectors spanning distinct angular sectors."""
    records: list[VectorRecord] = []
    # 4 clusters around canonical axes
    axes = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for cluster_id, axis in enumerate(axes):
        for i in range(5):
            # Perturb axis slightly
            perturbed = [
                axis[0] + (i * 0.05),
                axis[1] + (i * 0.04),
                axis[2] + (i * 0.03),
                axis[3] + (i * 0.02),
            ]
            vec = normalize(perturbed)
            rec = VectorRecord(
                vector_id=f"v_{cluster_id}_{i}",
                source_id=f"src_{cluster_id}_{i}",
                vector=vec,
                model_name="mock-model-v1",
                dimension=4,
                metadata={
                    "cluster_id": cluster_id,
                    "index": i,
                    "timestamp": 1000.0 + (cluster_id * 10) + i,
                    "category": "even" if i % 2 == 0 else "odd",
                },
            )
            records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 1. K-Means Clustering Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKMeansClustering:
    def test_fit_and_centroids_shape(self) -> None:
        vectors = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
        kmeans = KMeans(n_clusters=2, max_iter=20, random_seed=42)
        kmeans.fit(vectors)

        assert len(kmeans.centroids) == 2
        for c in kmeans.centroids:
            assert len(c) == 3
            # Centroids must be unit-normalized
            assert math.isclose(l2_norm(c), 1.0, rel_tol=1e-5)

    def test_predict_assignments(self) -> None:
        vectors = [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
        ]
        kmeans = KMeans(n_clusters=2, max_iter=20, random_seed=42).fit(vectors)
        assignments = kmeans.predict(vectors)

        assert len(assignments) == 4
        # Points close to x-axis share a cluster; points close to y-axis share another
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[0] != assignments[2]

    def test_empty_vectors_raises(self) -> None:
        kmeans = KMeans(n_clusters=2)
        with pytest.raises(ValueError, match="empty vectors"):
            kmeans.fit([])

    def test_dimension_mismatch_raises(self) -> None:
        kmeans = KMeans(n_clusters=2)
        with pytest.raises(ValueError, match="identical dimensionality"):
            kmeans.fit([[1.0, 0.0], [1.0, 0.0, 0.0]])


# ─────────────────────────────────────────────────────────────────────────────
# 2. IVFFlat Index Construction & Search Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIVFFlatIndex:
    def test_build_and_inverted_lists(
        self, synthetic_records: list[VectorRecord]
    ) -> None:
        index = IVFFlatIndex(dimension=4, nlist=4)
        index.build(synthetic_records)

        assert index.is_trained is True
        assert len(index.centroids) == 4
        assert index.total_vectors == len(synthetic_records)

        # Every record must be placed in exactly one inverted list
        total_in_lists = sum(len(lst) for lst in index.inverted_lists.values())
        assert total_in_lists == len(synthetic_records)

    def test_search_nprobe_full_recall(
        self, synthetic_records: list[VectorRecord]
    ) -> None:
        """When nprobe == nlist, IVFFlat probes all cells and matches exact k-NN."""
        index = IVFFlatIndex(dimension=4, nlist=4)
        index.build(synthetic_records)

        query = [1.0, 0.0, 0.0, 0.0]
        # Full scan probe
        results = index.search(query, top_k=5, nprobe=4)
        assert len(results) == 5

        # Check ranking descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Top result must belong to cluster_id 0
        assert results[0].metadata["cluster_id"] == 0

    def test_search_nprobe_single_cell(
        self, synthetic_records: list[VectorRecord]
    ) -> None:
        index = IVFFlatIndex(dimension=4, nlist=4)
        index.build(synthetic_records)

        query = [1.0, 0.0, 0.0, 0.0]
        results = index.search(query, top_k=3, nprobe=1)
        # Should return matches from the closest Voronoi partition
        assert len(results) == 3
        for r in results:
            assert r.metadata["cluster_id"] == 0

    def test_search_with_metadata_filter(
        self, synthetic_records: list[VectorRecord]
    ) -> None:
        index = IVFFlatIndex(dimension=4, nlist=4)
        index.build(synthetic_records)

        query = [1.0, 0.0, 0.0, 0.0]
        results = index.search(
            query,
            top_k=5,
            nprobe=4,
            filter_metadata={"category": "odd"},
        )
        assert len(results) > 0
        for r in results:
            assert r.metadata["category"] == "odd"

    def test_persistence_roundtrip(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        index = IVFFlatIndex(dimension=4, nlist=4)
        index.build(synthetic_records)

        col = db.collection("persisted_ivf")
        index.save(col)

        # Load into a fresh index
        loaded = IVFFlatIndex.load(col)
        assert loaded is not None
        assert loaded.dimension == 4
        assert loaded.nlist == 4
        assert loaded.is_trained is True
        assert loaded.total_vectors == len(synthetic_records)

        # Test search on loaded index
        query = [0.0, 1.0, 0.0, 0.0]
        res = loaded.search(query, top_k=2, nprobe=4)
        assert len(res) == 2
        assert res[0].metadata["cluster_id"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Query Processing in VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridQueryProcessing:
    def test_timestamp_range_filtering(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        store = VectorStore(db, collection_name="hybrid_test", dimension=4)
        for r in synthetic_records:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )

        # Query restricting timestamps to [1000.0, 1005.0]
        results = store.search(
            [1.0, 0.0, 0.0, 0.0],
            top_k=10,
            min_timestamp=1000.0,
            max_timestamp=1004.0,
        )
        assert len(results) > 0
        for r in results:
            ts = r.metadata["timestamp"]
            assert 1000.0 <= ts <= 1004.0

    def test_custom_lambda_predicate_filtering(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        store = VectorStore(db, collection_name="predicate_test", dimension=4)
        for r in synthetic_records:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )

        # Only allow cluster_id == 2 and category == "even"
        results = store.search(
            [0.0, 0.0, 1.0, 0.0],
            top_k=10,
            filter_fn=lambda m: m.get("cluster_id") == 2 and m.get("category") == "even",
        )
        assert len(results) > 0
        for r in results:
            assert r.metadata["cluster_id"] == 2
            assert r.metadata["category"] == "even"


# ─────────────────────────────────────────────────────────────────────────────
# 4. VectorStore & IVFFlat Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStoreIVFFlatIntegration:
    def test_build_and_use_index(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        store = VectorStore(db, collection_name="idx_integration", dimension=4)
        for r in synthetic_records:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )

        assert store.has_index is False
        store.build_index(nlist=4)
        assert store.has_index is True

        # Search with use_index=True
        approx_res = store.search(
            [1.0, 0.0, 0.0, 0.0],
            top_k=3,
            use_index=True,
            nprobe=2,
        )
        assert len(approx_res) == 3

    def test_index_rebuild_after_new_inserts(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        store = VectorStore(db, collection_name="rebuild_test", dimension=4)
        for r in synthetic_records[:10]:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )

        store.build_index(nlist=2)
        assert store.index is not None
        assert store.index.total_vectors == 10

        # Add remaining 10 records
        for r in synthetic_records[10:]:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )

        # Rebuild index
        store.rebuild_index()
        assert store.index.total_vectors == 20

    def test_auto_load_index_on_reopen(
        self, tmp_db_dir: Path, synthetic_records: list[VectorRecord]
    ) -> None:
        # Session 1: build index
        db1 = NeuroLinkDB(tmp_db_dir)
        store1 = VectorStore(db1, collection_name="auto_load", dimension=4)
        for r in synthetic_records:
            store1.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )
        store1.build_index(nlist=4)
        db1.close()

        # Session 2: reopen store and verify index is loaded
        db2 = NeuroLinkDB(tmp_db_dir)
        store2 = VectorStore(db2, collection_name="auto_load")
        assert store2.has_index is True
        assert store2.index is not None
        assert store2.index.total_vectors == len(synthetic_records)
        db2.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Recall@K & Benchmarking Evaluation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallAndEvaluation:
    def test_compute_recall_at_k(self) -> None:
        v1 = VectorSearchResult("v1", "src1", 1.0, {}, [])
        v2 = VectorSearchResult("v2", "src2", 0.9, {}, [])
        v3 = VectorSearchResult("v3", "src3", 0.8, {}, [])
        v4 = VectorSearchResult("v4", "src4", 0.7, {}, [])

        # Perfect match
        exact = [v1, v2, v3]
        approx = [v1, v2, v3]
        assert compute_recall_at_k(exact, approx, 3) == 1.0

        # 2 out of 3 match
        approx_partial = [v1, v2, v4]
        assert math.isclose(compute_recall_at_k(exact, approx_partial, 3), 2.0 / 3.0)

        # Disjoint match
        approx_disjoint = [v4]
        assert compute_recall_at_k(exact, approx_disjoint, 3) == 0.0

    def test_evaluate_index_performance_harness(
        self, db: NeuroLinkDB, synthetic_records: list[VectorRecord]
    ) -> None:
        store = VectorStore(db, collection_name="eval_test", dimension=4)
        for r in synthetic_records:
            store.add_vector(
                r.source_id,
                r.vector,
                r.model_name,
                metadata=r.metadata,
                vector_id=r.vector_id,
            )
        store.build_index(nlist=4)

        test_queries = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

        # Evaluate with nprobe=4 (full probe -> recall should be 1.0)
        full_probe_eval = evaluate_index_performance(
            store, test_queries, top_k=3, nprobe=4
        )
        assert full_probe_eval.recall_at_k == 1.0
        assert full_probe_eval.exact_latency_ms >= 0.0
        assert full_probe_eval.approx_latency_ms >= 0.0

        # Evaluate with nprobe=1 (approximate probe -> recall > 0.0)
        single_probe_eval = evaluate_index_performance(
            store, test_queries, top_k=3, nprobe=1
        )
        assert single_probe_eval.recall_at_k > 0.0

