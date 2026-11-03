"""
storage/conversation_store.py
──────────────────────────────
Conversation persistence layer backed by NeuroLinkDB.

Stores conversation turns with metadata:
  • conversation_id: Groups messages in a session
  • message_id: Unique identifier for each turn
  • role: 'user', 'assistant', or 'system'
  • content: Text content of the turn
  • timestamp: UNIX timestamp (float)
  • embedding_ref: Optional pointer to embedding vector (Phase 4 readiness)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.neurolinkdb.collection import Collection
from neurochat.llm.base import Message
from neurochat.embeddings.base import BaseEmbeddingProvider
from neurochat.storage.vector_store import VectorStore


@dataclass
class StoredMessage:
    """
    Representation of a single message persisted in NeuroLinkDB.

    Attributes
    ----------
    conversation_id:
        Identifier grouping messages belonging to the same chat thread.
    message_id:
        Unique UUID or string identifying this particular message turn.
    role:
        Role of the speaker: 'user', 'assistant', or 'system'.
    content:
        Message text.
    timestamp:
        UNIX timestamp when the message was recorded.
    embedding_ref:
        Optional reference identifier to an embedding record (for Phase 4+).
    """

    conversation_id: str
    message_id: str
    role: str
    content: str
    timestamp: float
    embedding_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredMessage":
        """Instantiate a StoredMessage from a dictionary."""
        return cls(
            conversation_id=str(data["conversation_id"]),
            message_id=str(data["message_id"]),
            role=str(data["role"]),
            content=str(data["content"]),
            timestamp=float(data.get("timestamp", time.time())),
            embedding_ref=data.get("embedding_ref"),
        )

    def to_message(self) -> Message:
        """Convert into a core LLM Message instance."""
        return Message(role=self.role, content=self.content)


class ConversationStore:
    """
    Manages conversation persistence using a NeuroLinkDB collection.

    Parameters
    ----------
    db:
        An open NeuroLinkDB instance.
    collection_name:
        Name of the collection to store message records in (default: 'messages').
    """

    DEFAULT_COLLECTION_NAME: str = "messages"

    def __init__(
        self,
        db: NeuroLinkDB,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._collection: Collection = db.collection(collection_name)
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider

    @property
    def vector_store(self) -> Optional[VectorStore]:
        return self._vector_store

    @property
    def embedding_provider(self) -> Optional[BaseEmbeddingProvider]:
        return self._embedding_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        embedding_ref: Optional[str] = None,
    ) -> StoredMessage:
        """
        Record a new message turn into the database.

        If a vector store and embedding provider are attached, an embedding
        vector is automatically generated and linked.
        """
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty.")
        if not role.strip():
            raise ValueError("role must not be empty.")
        if not content.strip():
            raise ValueError("content must not be empty.")

        msg_id = message_id or str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        final_embedding_ref = embedding_ref

        # Generate and store embedding vector if components are configured
        if self._vector_store and self._embedding_provider and final_embedding_ref is None:
            vec = self._embedding_provider.embed_text(content.strip())
            v_rec = self._vector_store.add_vector(
                source_id=msg_id,
                vector=vec,
                model_name=self._embedding_provider.model_name,
                metadata={
                    "conversation_id": conversation_id.strip(),
                    "role": role.strip(),
                },
            )
            final_embedding_ref = v_rec.vector_id

        stored_msg = StoredMessage(
            conversation_id=conversation_id.strip(),
            message_id=msg_id,
            role=role.strip(),
            content=content.strip(),
            timestamp=ts,
            embedding_ref=final_embedding_ref,
        )

        key = self._format_key(conversation_id.strip(), msg_id)
        self._collection.insert(key, stored_msg.to_dict())
        return stored_msg

    def get_messages(self, conversation_id: str) -> List[StoredMessage]:
        """
        Retrieve all non-deleted messages for *conversation_id* in chronological order.

        Parameters
        ----------
        conversation_id:
            Target conversation session ID.

        Returns
        -------
        List[StoredMessage]
            Sorted list of messages by timestamp (and insertion order if equal).
        """
        target_id = conversation_id.strip()
        matched: List[tuple[int, StoredMessage]] = []

        for record in self._collection.scan():
            val = record.value
            if isinstance(val, dict) and val.get("conversation_id") == target_id:
                try:
                    msg = StoredMessage.from_dict(val)
                    matched.append((record.record_id, msg))
                except (KeyError, ValueError, TypeError):
                    continue

        # Sort primarily by timestamp, secondarily by record_id
        matched.sort(key=lambda item: (item[1].timestamp, item[0]))
        return [item[1] for item in matched]

    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List distinct conversations with summary metadata.

        Returns
        -------
        List[Dict[str, Any]]
            List of conversation summaries sorted by most recent activity:
            {
                "conversation_id": str,
                "message_count": int,
                "first_timestamp": float,
                "last_timestamp": float,
                "last_message": str,
            }
        """
        groups: Dict[str, List[StoredMessage]] = {}

        for record in self._collection.scan():
            val = record.value
            if isinstance(val, dict) and "conversation_id" in val:
                try:
                    msg = StoredMessage.from_dict(val)
                    groups.setdefault(msg.conversation_id, []).append(msg)
                except (KeyError, ValueError, TypeError):
                    continue

        summaries: List[Dict[str, Any]] = []
        for cid, msgs in groups.items():
            if not msgs:
                continue
            msgs.sort(key=lambda m: m.timestamp)
            summaries.append(
                {
                    "conversation_id": cid,
                    "message_count": len(msgs),
                    "first_timestamp": msgs[0].timestamp,
                    "last_timestamp": msgs[-1].timestamp,
                    "last_message": msgs[-1].content,
                }
            )

        # Most recently active conversations first
        summaries.sort(key=lambda s: s["last_timestamp"], reverse=True)
        return summaries

    def delete_conversation(self, conversation_id: str) -> int:
        """
        Tombstone-delete all messages belonging to *conversation_id*.

        Parameters
        ----------
        conversation_id:
            Target conversation session ID.

        Returns
        -------
        int
            Number of message records marked deleted.
        """
        target_id = conversation_id.strip()
        keys_to_delete: List[str] = []

        for record in self._collection.scan():
            val = record.value
            if isinstance(val, dict) and val.get("conversation_id") == target_id:
                keys_to_delete.append(record.key)

        deleted_count = 0
        for k in keys_to_delete:
            deleted_count += self._collection.delete(k)

        # Also purge vectors linked to this conversation if vector_store is attached
        if self._vector_store:
            for rec in self._vector_store._collection.scan():
                if isinstance(rec.value, dict):
                    m = rec.value.get("metadata", {})
                    if m.get("conversation_id") == target_id:
                        self._vector_store._collection.delete(rec.key)

        return deleted_count

    def search_similar_messages(
        self,
        query_text: str,
        *,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[StoredMessage, float]]:
        """
        Retrieve messages most semantically similar to *query_text*.

        Parameters
        ----------
        query_text:
            Search prompt or user message.
        conversation_id:
            Optional filter to restrict retrieval to a specific conversation thread.
        top_k:
            Maximum number of top-ranked messages to return.
        threshold:
            Minimum cosine similarity threshold.

        Returns
        -------
        List[Tuple[StoredMessage, float]]
            List of (StoredMessage, score) pairs sorted descending by score.
        """
        if not self._vector_store or not self._embedding_provider:
            return []

        query_vec = self._embedding_provider.embed_text(query_text)
        filter_meta = {"conversation_id": conversation_id.strip()} if conversation_id else None

        vector_results = self._vector_store.search(
            query_vec,
            top_k=top_k,
            threshold=threshold,
            filter_metadata=filter_meta,
        )

        if not vector_results:
            return []

        # Map source_id (message_id) to similarity score
        score_by_msg_id = {vr.source_id: vr.score for vr in vector_results}

        # Retrieve full StoredMessage records
        found_messages: Dict[str, StoredMessage] = {}
        for rec in self._collection.scan():
            val = rec.value
            if isinstance(val, dict):
                m_id = val.get("message_id")
                if m_id in score_by_msg_id and m_id not in found_messages:
                    try:
                        found_messages[m_id] = StoredMessage.from_dict(val)
                    except (KeyError, ValueError, TypeError):
                        pass

        # Package results preserving vector rank order
        ordered_results: List[Tuple[StoredMessage, float]] = []
        for vr in vector_results:
            if vr.source_id in found_messages:
                ordered_results.append((found_messages[vr.source_id], vr.score))

        return ordered_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_key(conversation_id: str, message_id: str) -> str:
        """Generate a deterministic storage key for a message."""
        return f"msg:{conversation_id}:{message_id}"

