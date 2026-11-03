"""
storage/__init__.py
───────────────────
Public exports for the conversation persistence layer.
"""

from neurochat.storage.conversation_store import ConversationStore, StoredMessage
from neurochat.storage.vector_store import (
    VectorStore,
    VectorRecord,
    VectorSearchResult,
    IncompatibleVectorError,
)

__all__ = [
    "ConversationStore",
    "StoredMessage",
    "VectorStore",
    "VectorRecord",
    "VectorSearchResult",
    "IncompatibleVectorError",
]

