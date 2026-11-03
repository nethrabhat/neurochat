"""
NeuroChat document ingestion package.

Provides:
  - TextSplitter: chunk plain text with configurable overlap
  - DocumentStore: persist, embed, and search ingested document chunks
"""

from neurochat.ingestion.splitter import TextSplitter
from neurochat.ingestion.document_store import DocumentRecord, DocumentStore

__all__ = ["TextSplitter", "DocumentRecord", "DocumentStore"]

