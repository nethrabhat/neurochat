"""
tests/test_embeddings.py
─────────────────────────
Phase 4 test suite: Embeddings, Vector Math, and Exact k-NN Vector Search.

Covers:
  - Vector math: dot_product, l2_norm, normalize, cosine_similarity, euclidean_distance
  - Exact mathematical correctness on known synthetic vectors
  - Dimension mismatch and zero-norm safety
  - MockEmbeddingProvider determinism and normalization
  - VectorStore insertion, dimension validation, and model constraint checks
  - Exact k-NN search ranking against synthetic ground truth
  - Top-k and similarity threshold filtering
  - Metadata filtering for hybrid queries
  - Persistence across database closing and reopening
  - Integration with ConversationStore (auto-embedding and semantic message search)

All tests run fully offline with zero external network or GPU requirements.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from neurochat.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_norm,
    normalize,
    validate_equal_dimensions,
)
from neurochat.embeddings.base import BaseEmbeddingProvider, MockEmbeddingProvider
from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.storage.vector_store import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
    IncompatibleVectorError,
)
from neurochat.storage.conversation_store import ConversationStore, StoredMessage


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    return tmp_path / "vec_db"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=8, model_name="test-embed-v1")


@pytest.fixture
def vector_store(db: NeuroLinkDB) -> VectorStore:
    return VectorStore(db, dimension=4, model_name="test-embed-v1")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Math & Similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorMath:
    def test_dot_product_orthogonal(self) -> None:
        assert dot_product([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_dot_product_parallel(self) -> None:
        assert dot_product([2.0, 3.0], [4.0, 5.0]) == 23.0

    def test_l2_norm(self) -> None:
        assert math.isclose(l2_norm([3.0, 4.0]), 5.0)

    def test_normalize_unit_length(self) -> None:
        vec = [3.0, 4.0]
        normed = normalize(vec)
        assert math.isclose(l2_norm(normed), 1.0)
        assert math.isclose(normed[0], 0.6)
        assert math.isclose(normed[1], 0.8)

    def test_normalize_zero_vector(self) -> None:
        normed = normalize([0.0, 0.0, 0.0])
        assert normed == [0.0, 0.0, 0.0]

    def test_euclidean_distance(self) -> None:
        dist = euclidean_distance([1.0, 2.0], [4.0, 6.0])
        assert math.isclose(dist, 5.0)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        assert math.isclose(cosine_similarity([2.0, 5.0, 8.0], [2.0, 5.0, 8.0]), 1.0)

    def test_orthogonal_vectors(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0)

    def test_opposite_vectors(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_known_45_degree_angle(self) -> None:
        # Angle between [1, 0] and [1, 1] is 45 degrees; cos(45 deg) = 1/sqrt(2)
        score = cosine_similarity([1.0, 0.0], [1.0, 1.0])
        assert math.isclose(score, 1.0 / math.sqrt(2.0))

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            cosine_similarity([], [])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Embedding Providers
# ─────────────────────────────────────────────────────────────────────────────

class TestMockEmbeddingProvider:
    def test_provider_properties(self, mock_embedder: MockEmbeddingProvider) -> None:
        assert mock_embedder.dimension == 8
        assert mock_embedder.model_name == "test-embed-v1"

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            MockEmbeddingProvider(dimension=0)

    def test_determinism(self, mock_embedder: MockEmbeddingProvider) -> None:
        v1 = mock_embedder.embed_text("database systems internals")
        v2 = mock_embedder.embed_text("database systems internals")
        assert v1 == v2

    def test_unit_normalization(self, mock_embedder: MockEmbeddingProvider) -> None:
        vec = mock_embedder.embed_text("hello world from test")
        assert len(vec) == 8
        assert math.isclose(l2_norm(vec), 1.0, rel_tol=1e-5)

    def test_batch_embedding(self, mock_embedder: MockEmbeddingProvider) -> None:
        batch = mock_embedder.embed_batch(["alpha", "beta", "gamma"])
        assert len(batch) == 3
        for v in batch:
            assert len(v) == 8
            assert math.isclose(l2_norm(v), 1.0, rel_tol=1e-5)

    def test_semantic_word_overlap_similarity(
        self, mock_embedder: MockEmbeddingProvider
    ) -> None:
        # Embed texts with shared vs distinct vocabulary
        v_db1 = mock_embedder.embed_text("database systems storage")
        v_db2 = mock_embedder.embed_text("database systems query engine")
        v_other = mock_embedder.embed_text("cooking lasagna with marinara")

        sim_related = cosine_similarity(v_db1, v_db2)
        sim_unrelated = cosine_similarity(v_db1, v_other)

        assert sim_related > sim_unrelated


# ─────────────────────────────────────────────────────────────────────────────
# 3. VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_get_vector(self, vector_store: VectorStore) -> None:
        rec = vector_store.add_vector(
            source_id="msg:1",
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="test-embed-v1",
            metadata={"tag": "demo"},
        )
        assert rec.source_id == "msg:1"
        assert rec.dimension == 4

        fetched = vector_store.get_vector(rec.vector_id)
        assert fetched is not None
        assert fetched.vector_id == rec.vector_id
        assert fetched.vector == [1.0, 0.0, 0.0, 0.0]
        assert fetched.metadata == {"tag": "demo"}

    def test_reject_incompatible_dimension(self, vector_store: VectorStore) -> None:
        # Collection expects dimension=4; passing 3 should raise
        with pytest.raises(IncompatibleVectorError, match="dimension"):
            vector_store.add_vector(
                source_id="m",
                vector=[1.0, 2.0, 3.0],
                model_name="test-embed-v1",
            )

    def test_reject_incompatible_model(self, vector_store: VectorStore) -> None:
        # Collection expects model 'test-embed-v1'
        with pytest.raises(IncompatibleVectorError, match="model"):
            vector_store.add_vector(
                source_id="m",
                vector=[1.0, 0.0, 0.0, 0.0],
                model_name="unsupported-other-model",
            )

    def test_dimension_locked_on_first_insert_when_none(
        self, db: NeuroLinkDB
    ) -> None:
        store = VectorStore(db, collection_name="dynamic_vecs")
        assert store.dimension is None

        store.add_vector(
            source_id="s1",
            vector=[0.1, 0.2, 0.3],
            model_name="dynamic-model",
        )
        assert store.dimension == 3
        assert store.model_name == "dynamic-model"

        # Subsequent insert with dimension=4 must fail
        with pytest.raises(IncompatibleVectorError, match="dimension"):
            store.add_vector(
                source_id="s2",
                vector=[0.1, 0.2, 0.3, 0.4],
                model_name="dynamic-model",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exact Nearest-Neighbor Search (Ground Truth Verification)
# ─────────────────────────────────────────────────────────────────────────────

class TestExactKNN:
    def test_exact_ranking_ground_truth(self, vector_store: VectorStore) -> None:
        """
        Populate with synthetic vectors of known angles to query [1, 0, 0, 0]:
          Target A: [1, 0, 0, 0]        -> cosine similarity = 1.000
          Target B: [0.8, 0.6, 0, 0]    -> cosine similarity = 0.800
          Target C: [0.5, 0.5, 0.5, 0.5] -> cosine similarity = 0.500
          Target D: [0, 1, 0, 0]        -> cosine similarity = 0.000
          Target E: [-1, 0, 0, 0]       -> cosine similarity = -1.000
        """
        vector_store.add_vector("A", [1.0, 0.0, 0.0, 0.0], "test-embed-v1")
        vector_store.add_vector("B", [0.8, 0.6, 0.0, 0.0], "test-embed-v1")
        vector_store.add_vector("C", [0.5, 0.5, 0.5, 0.5], "test-embed-v1")
        vector_store.add_vector("D", [0.0, 1.0, 0.0, 0.0], "test-embed-v1")
        vector_store.add_vector("E", [-1.0, 0.0, 0.0, 0.0], "test-embed-v1")

        query = [1.0, 0.0, 0.0, 0.0]
        results = vector_store.search(query, top_k=5)

        assert len(results) == 5
        source_order = [r.source_id for r in results]
        assert source_order == ["A", "B", "C", "D", "E"]

        assert math.isclose(results[0].score, 1.0)
        assert math.isclose(results[1].score, 0.8)
        assert math.isclose(results[2].score, 0.5)
        assert math.isclose(results[3].score, 0.0)
        assert math.isclose(results[4].score, -1.0)

    def test_search_top_k_limit(self, vector_store: VectorStore) -> None:
        for i in range(10):
            vector_store.add_vector(f"s{i}", [float(i), 1.0, 0.0, 0.0], "test-embed-v1")

        results = vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_threshold_filtering(self, vector_store: VectorStore) -> None:
        vector_store.add_vector("high", [1.0, 0.0, 0.0, 0.0], "test-embed-v1")
        vector_store.add_vector("medium", [0.5, 0.5, 0.5, 0.5], "test-embed-v1")
        vector_store.add_vector("low", [-1.0, 0.0, 0.0, 0.0], "test-embed-v1")

        # Threshold = 0.6 should only keep 'high' (1.0)
        filtered = vector_store.search([1.0, 0.0, 0.0, 0.0], threshold=0.6)
        assert len(filtered) == 1
        assert filtered[0].source_id == "high"

    def test_search_metadata_filtering(self, vector_store: VectorStore) -> None:
        vector_store.add_vector(
            "doc1",
            [1.0, 0.0, 0.0, 0.0],
            "test-embed-v1",
            metadata={"category": "tech"},
        )
        vector_store.add_vector(
            "doc2",
            [0.9, 0.1, 0.0, 0.0],
            "test-embed-v1",
            metadata={"category": "cooking"},
        )

        # Restrict to cooking
        results = vector_store.search(
            [1.0, 0.0, 0.0, 0.0],
            filter_metadata={"category": "cooking"},
        )
        assert len(results) == 1
        assert results[0].source_id == "doc2"

    def test_delete_by_source_id(self, vector_store: VectorStore) -> None:
        vector_store.add_vector("doc-del", [1.0, 0.0, 0.0, 0.0], "test-embed-v1")
        vector_store.add_vector("doc-keep", [0.0, 1.0, 0.0, 0.0], "test-embed-v1")

        assert vector_store.count() == 2
        del_count = vector_store.delete_by_source_id("doc-del")
        assert del_count == 1
        assert vector_store.count() == 1

        remaining = vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(remaining) == 1
        assert remaining[0].source_id == "doc-keep"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Persistence Across DB Reopen
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorPersistence:
    def test_vectors_survive_reopen(self, tmp_db_dir: Path) -> None:
        # Session 1: insert vectors
        db1 = NeuroLinkDB(tmp_db_dir)
        vs1 = VectorStore(db1, dimension=3, model_name="persist-v1")
        vs1.add_vector("item:1", [0.6, 0.8, 0.0], "persist-v1", metadata={"key": "val"})
        db1.close()

        # Session 2: reopen
        db2 = NeuroLinkDB(tmp_db_dir)
        vs2 = VectorStore(db2)
        assert vs2.dimension == 3
        assert vs2.model_name == "persist-v1"
        assert vs2.count() == 1

        matches = vs2.search([1.0, 0.0, 0.0], top_k=1)
        assert len(matches) == 1
        assert matches[0].source_id == "item:1"
        assert matches[0].metadata == {"key": "val"}
        assert math.isclose(matches[0].score, 0.6)
        db2.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: ConversationStore Semantic Search
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationSemanticIntegration:
    def test_auto_embedding_and_semantic_search(
        self, db: NeuroLinkDB, mock_embedder: MockEmbeddingProvider
    ) -> None:
        v_store = VectorStore(
            db,
            collection_name="chat_vectors",
            dimension=mock_embedder.dimension,
            model_name=mock_embedder.model_name,
        )
        c_store = ConversationStore(
            db,
            vector_store=v_store,
            embedding_provider=mock_embedder,
        )

        # Insert messages with distinct topics
        m1 = c_store.add_message(
            "conv-1",
            "user",
            "How does database storage engine design work?",
        )
        m2 = c_store.add_message(
            "conv-1",
            "assistant",
            "Database storage engines use binary pages and file allocation.",
        )
        m3 = c_store.add_message(
            "conv-1",
            "user",
            "What is the best recipe for baking Italian chocolate cake?",
        )

        # Verify embedding references were populated
        assert m1.embedding_ref is not None
        assert m2.embedding_ref is not None
        assert m3.embedding_ref is not None
        assert v_store.count() == 3

        # Semantic query about database internals
        results = c_store.search_similar_messages(
            "database binary pages storage",
            conversation_id="conv-1",
            top_k=2,
        )
        assert len(results) == 2
        # The top 2 returned messages should be about databases, not cake
        contents = [msg.content for msg, score in results]
        assert m3.content not in contents

    def test_delete_conversation_cleans_vectors(
        self, db: NeuroLinkDB, mock_embedder: MockEmbeddingProvider
    ) -> None:
        v_store = VectorStore(
            db,
            collection_name="clean_vectors",
            dimension=mock_embedder.dimension,
            model_name=mock_embedder.model_name,
        )
        c_store = ConversationStore(
            db,
            vector_store=v_store,
            embedding_provider=mock_embedder,
        )

        c_store.add_message("del-thread", "user", "Message to be purged")
        c_store.add_message("keep-thread", "user", "Message to be retained")

        assert v_store.count() == 2
        c_store.delete_conversation("del-thread")

        assert v_store.count() == 1
        remaining = v_store.search(
            mock_embedder.embed_text("query"), top_k=5
        )
        assert len(remaining) == 1
        assert remaining[0].metadata.get("conversation_id") == "keep-thread"

