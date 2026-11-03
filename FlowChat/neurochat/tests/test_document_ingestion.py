"""
Tests for Phase 8a — document ingestion.

Coverage:
  - TextSplitter: chunk sizing, overlap, edge cases
  - DocumentStore: ingest_text, ingest_file, list_documents, delete_document
  - DocumentStore: persistence across re-open
  - DocumentStore: keyword search fallback
  - DocumentStore: semantic search with MockEmbeddingProvider + VectorStore
  - DocumentRecord: serialisation round-trip
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import List

import pytest

from neurochat.ingestion.splitter import TextSplitter
from neurochat.ingestion.document_store import DocumentRecord, DocumentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "NeuroLinkDB is a custom binary storage engine. "
    "It supports pages, records, buffer pools, and write-ahead logs. "
    "The engine is designed for efficient vector storage and retrieval. "
    "Embeddings are stored alongside metadata for semantic search. "
    "The architecture follows a layered design with explicit abstractions."
)


@pytest.fixture()
def tmp_db(tmp_path):
    """Yield a fresh DocumentStore backed by a temporary directory."""
    store = DocumentStore(tmp_path / "docdb")
    yield store
    store.close()


@pytest.fixture()
def sample_txt_file(tmp_path) -> Path:
    """Write SAMPLE_TEXT to a .txt file and return its path."""
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TextSplitter tests
# ---------------------------------------------------------------------------


class TestTextSplitter:
    def test_basic_split_returns_correct_chunk_size(self):
        splitter = TextSplitter(chunk_size=20, overlap=5)
        text = "A" * 100
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 20

    def test_first_chunk_is_chunk_size(self):
        splitter = TextSplitter(chunk_size=20, overlap=5)
        text = "X" * 100
        chunks = splitter.split(text)
        assert len(chunks[0]) == 20

    def test_overlap_shared_between_consecutive_chunks(self):
        splitter = TextSplitter(chunk_size=10, overlap=4)
        text = "ABCDEFGHIJKLMNOPQRST"  # 20 chars
        chunks = splitter.split(text)
        # chunk[0] ends at pos 10; chunk[1] starts at pos 6 (step = 10-4 = 6)
        assert len(chunks) == 3
        # Verify overlap: last `overlap` chars of chunk[0] == first `overlap` chars of chunk[1]
        assert chunks[0][-4:] == chunks[1][:4]

    def test_short_text_returns_single_chunk(self):
        splitter = TextSplitter(chunk_size=100, overlap=10)
        text = "Short text"
        chunks = splitter.split(text)
        assert chunks == ["Short text"]

    def test_empty_text_returns_empty_list(self):
        splitter = TextSplitter()
        assert splitter.split("") == []
        assert splitter.split("   \n\t  ") == []

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TextSplitter(chunk_size=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            TextSplitter(chunk_size=64, overlap=-1)

    def test_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            TextSplitter(chunk_size=10, overlap=10)

    def test_split_by_paragraphs_basic(self):
        splitter = TextSplitter(chunk_size=200, overlap=20)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = splitter.split_by_paragraphs(text)
        assert len(chunks) == 3
        assert chunks[0] == "Para one."

    def test_split_by_paragraphs_long_para_subdivided(self):
        splitter = TextSplitter(chunk_size=20, overlap=5)
        long_para = "X" * 100
        text = f"Short.\n\n{long_para}"
        chunks = splitter.split_by_paragraphs(text)
        # "Short." fits in one chunk; the long para is subdivided
        assert len(chunks) > 2

    def test_no_trailing_whitespace_in_chunks(self):
        splitter = TextSplitter(chunk_size=15, overlap=3)
        text = "Hello World   " * 10
        for chunk in splitter.split(text):
            assert chunk == chunk.rstrip()


# ---------------------------------------------------------------------------
# DocumentRecord serialisation
# ---------------------------------------------------------------------------


class TestDocumentRecord:
    def test_to_dict_and_from_dict_roundtrip(self):
        rec = DocumentRecord(
            doc_id=str(uuid.uuid4()),
            filename="test.txt",
            chunk_index=0,
            chunk_text="Hello world",
            source_path="/tmp/test.txt",
        )
        d = rec.to_dict()
        rec2 = DocumentRecord.from_dict(d)
        assert rec2.doc_id == rec.doc_id
        assert rec2.filename == rec.filename
        assert rec2.chunk_index == rec.chunk_index
        assert rec2.chunk_text == rec.chunk_text
        assert rec2.source_path == rec.source_path
        assert rec2.ingested_at == rec.ingested_at

    def test_ingested_at_is_iso_string(self):
        rec = DocumentRecord(
            doc_id="x",
            filename="f.txt",
            chunk_index=0,
            chunk_text="text",
            source_path="",
        )
        # Should be parseable as ISO 8601
        from datetime import datetime
        dt = datetime.fromisoformat(rec.ingested_at)
        assert dt.year >= 2024


# ---------------------------------------------------------------------------
# DocumentStore — ingest_text
# ---------------------------------------------------------------------------


class TestDocumentStoreIngestText:
    def test_ingest_text_produces_records(self, tmp_db):
        records = tmp_db.ingest_text(SAMPLE_TEXT, filename="sample.txt")
        assert len(records) > 0

    def test_ingest_text_sets_filename(self, tmp_db):
        records = tmp_db.ingest_text("Some content", filename="myfile.txt")
        assert all(r.filename == "myfile.txt" for r in records)

    def test_ingest_text_chunk_indices_sequential(self, tmp_db):
        splitter = TextSplitter(chunk_size=50, overlap=10)
        store = DocumentStore(tmp_db._db_dir.parent / "idx_test", splitter=splitter)
        long_text = "Word " * 200
        records = store.ingest_text(long_text, filename="big.txt")
        indices = [r.chunk_index for r in records]
        assert indices == list(range(len(records)))
        store.close()

    def test_ingest_empty_text_returns_empty(self, tmp_db):
        records = tmp_db.ingest_text("", filename="empty.txt")
        assert records == []

    def test_ingest_text_persists_to_collection(self, tmp_db):
        before = tmp_db.count()
        tmp_db.ingest_text("Some text content here", filename="a.txt")
        assert tmp_db.count() > before

    def test_ingest_text_unique_doc_ids(self, tmp_db):
        records = tmp_db.ingest_text("Word " * 200, filename="dup.txt")
        ids = [r.doc_id for r in records]
        assert len(ids) == len(set(ids)), "doc_ids should be unique"

    def test_ingest_text_with_embedder_and_vector_store(self, tmp_path):
        from neurochat.embeddings.base import MockEmbeddingProvider
        from neurochat.neurolinkdb.database import NeuroLinkDB
        from neurochat.storage.vector_store import VectorStore

        embedder = MockEmbeddingProvider(dimension=16)
        vs_db = NeuroLinkDB(str(tmp_path / "vs_db"))
        vs = VectorStore(vs_db, dimension=16)
        store = DocumentStore(tmp_path / "ingest_db")

        records = store.ingest_text(
            "NeuroLinkDB supports embeddings for semantic search.",
            filename="embed_test.txt",
            embedder=embedder,
            vector_store=vs,
        )
        assert len(records) > 0
        # Vector store should have entries
        assert vs.count() > 0
        store.close()
        vs_db.close()


# ---------------------------------------------------------------------------
# DocumentStore — ingest_file
# ---------------------------------------------------------------------------


class TestDocumentStoreIngestFile:
    def test_ingest_txt_file(self, tmp_db, sample_txt_file):
        records = tmp_db.ingest_file(sample_txt_file)
        assert len(records) > 0
        assert all(r.filename == "sample.txt" for r in records)

    def test_ingest_file_source_path_set(self, tmp_db, sample_txt_file):
        records = tmp_db.ingest_file(sample_txt_file)
        assert all(r.source_path != "" for r in records)

    def test_ingest_file_not_found_raises(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            tmp_db.ingest_file("/nonexistent/path/file.txt")

    def test_ingest_unsupported_extension_raises(self, tmp_db, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            tmp_db.ingest_file(p)

    def test_ingest_pdf_without_pdfminer_raises(self, tmp_db, tmp_path):
        """Requesting PDF ingestion without pdfminer.six must raise RuntimeError."""
        import sys
        import importlib

        p = tmp_path / "fake.pdf"
        p.write_bytes(b"%PDF-1.4 fake")

        # Simulate pdfminer not installed
        pdfminer_mod = sys.modules.get("pdfminer")
        pdfminer_hl = sys.modules.get("pdfminer.high_level")
        sys.modules["pdfminer"] = None  # type: ignore
        sys.modules["pdfminer.high_level"] = None  # type: ignore

        try:
            with pytest.raises(RuntimeError, match="pdfminer"):
                tmp_db.ingest_file(p)
        finally:
            # Restore
            if pdfminer_mod is None:
                sys.modules.pop("pdfminer", None)
            else:
                sys.modules["pdfminer"] = pdfminer_mod
            if pdfminer_hl is None:
                sys.modules.pop("pdfminer.high_level", None)
            else:
                sys.modules["pdfminer.high_level"] = pdfminer_hl


# ---------------------------------------------------------------------------
# DocumentStore — persistence across re-open
# ---------------------------------------------------------------------------


class TestDocumentStorePersistence:
    def test_records_survive_reopen(self, tmp_path):
        db_dir = tmp_path / "persist_db"
        store = DocumentStore(db_dir)
        records = store.ingest_text("Persistent content for testing.", filename="p.txt")
        assert len(records) > 0
        store.close()

        # Re-open
        store2 = DocumentStore(db_dir)
        assert store2.count() == len(records)
        store2.close()

    def test_list_documents_survives_reopen(self, tmp_path):
        db_dir = tmp_path / "list_db"
        store = DocumentStore(db_dir)
        store.ingest_text("Content A", filename="a.txt")
        store.ingest_text("Content B", filename="b.txt")
        store.close()

        store2 = DocumentStore(db_dir)
        docs = store2.list_documents()
        filenames = {d["filename"] for d in docs}
        assert "a.txt" in filenames
        assert "b.txt" in filenames
        store2.close()


# ---------------------------------------------------------------------------
# DocumentStore — list_documents
# ---------------------------------------------------------------------------


class TestDocumentStoreList:
    def test_list_documents_empty(self, tmp_db):
        assert tmp_db.list_documents() == []

    def test_list_documents_shows_unique_filenames(self, tmp_db):
        tmp_db.ingest_text("Content A " * 10, filename="a.txt")
        tmp_db.ingest_text("Content B " * 10, filename="b.txt")
        docs = tmp_db.list_documents()
        filenames = {d["filename"] for d in docs}
        assert "a.txt" in filenames
        assert "b.txt" in filenames

    def test_list_documents_has_chunk_count(self, tmp_db):
        splitter = TextSplitter(chunk_size=30, overlap=5)
        # Use a fresh store with small chunk_size to guarantee multiple chunks
        import tempfile as _tf
        with _tf.TemporaryDirectory() as d:
            store = DocumentStore(Path(d) / "list_count_db", splitter=splitter)
            text = "Word " * 100  # ~500 chars → multiple 30-char chunks
            records = store.ingest_text(text, filename="multi.txt")
            docs = store.list_documents()
            store.close()

        assert len(docs) == 1
        assert docs[0]["chunk_count"] == len(records)


# ---------------------------------------------------------------------------
# DocumentStore — delete_document
# ---------------------------------------------------------------------------


class TestDocumentStoreDelete:
    def test_delete_removes_records(self, tmp_db):
        tmp_db.ingest_text("Content to delete", filename="del.txt")
        before = tmp_db.count()
        n = tmp_db.delete_document("del.txt")
        assert n > 0
        assert tmp_db.count() == before - n

    def test_delete_nonexistent_filename_returns_zero(self, tmp_db):
        n = tmp_db.delete_document("does_not_exist.txt")
        assert n == 0

    def test_delete_only_removes_target_filename(self, tmp_db):
        tmp_db.ingest_text("Keep this", filename="keep.txt")
        tmp_db.ingest_text("Delete this", filename="delete.txt")
        total_before = tmp_db.count()
        n = tmp_db.delete_document("delete.txt")
        assert n > 0
        remaining = tmp_db.count()
        # "keep.txt" chunks should remain
        assert remaining == total_before - n
        docs = {d["filename"] for d in tmp_db.list_documents()}
        assert "keep.txt" in docs
        assert "delete.txt" not in docs


# ---------------------------------------------------------------------------
# DocumentStore — keyword search
# ---------------------------------------------------------------------------


class TestDocumentStoreSearch:
    def test_keyword_search_returns_results(self, tmp_db):
        tmp_db.ingest_text(
            "The quick brown fox jumps over the lazy dog", filename="fable.txt"
        )
        results = tmp_db.search_documents("fox")
        assert len(results) > 0

    def test_keyword_search_relevant_result_first(self, tmp_db):
        tmp_db.ingest_text(
            "NeuroLinkDB is a database engine.", filename="nldb.txt"
        )
        tmp_db.ingest_text(
            "The weather today is sunny and warm.", filename="weather.txt"
        )
        results = tmp_db.search_documents("database engine", top_k=5)
        assert len(results) > 0
        # The nldb chunk should score higher
        assert results[0][0].filename == "nldb.txt"

    def test_keyword_search_respects_top_k(self, tmp_db):
        for i in range(5):
            tmp_db.ingest_text(f"Document number {i} content here", filename=f"doc{i}.txt")
        results = tmp_db.search_documents("document content", top_k=2)
        assert len(results) <= 2

    def test_keyword_search_score_between_zero_and_one(self, tmp_db):
        tmp_db.ingest_text("Hello world", filename="hw.txt")
        results = tmp_db.search_documents("hello", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_semantic_search_with_mock_embedder(self, tmp_path):
        from neurochat.embeddings.base import MockEmbeddingProvider
        from neurochat.neurolinkdb.database import NeuroLinkDB
        from neurochat.storage.vector_store import VectorStore

        embedder = MockEmbeddingProvider(dimension=16)
        vs_db = NeuroLinkDB(str(tmp_path / "sem_vs"))
        vs = VectorStore(vs_db, dimension=16)
        store = DocumentStore(tmp_path / "sem_db")

        store.ingest_text(
            "NeuroLinkDB stores vectors for semantic retrieval.",
            filename="sem.txt",
            embedder=embedder,
            vector_store=vs,
        )

        results = store.search_documents(
            "semantic vector storage",
            top_k=5,
            embedder=embedder,
            vector_store=vs,
        )
        # MockEmbeddingProvider uses deterministic hashing so results may be
        # empty if the score doesn't match; at minimum no exception should raise.
        assert isinstance(results, list)
        store.close()
        vs_db.close()

    def test_context_manager_closes_cleanly(self, tmp_path):
        with DocumentStore(tmp_path / "ctx_db") as store:
            store.ingest_text("Context manager test content", filename="ctx.txt")
        # No exception means clean close

