"""
services/retrieval_service.py
─────────────────────────────
Context-aware retrieval service for NeuroChat.

Determines when context retrieval is appropriate, queries NeuroLinkDB for
semantically relevant historical messages, bounds the context size, and
formats the retrieved memory for injection into the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from neurochat.storage.conversation_store import ConversationStore, StoredMessage


@dataclass
class RetrievedItem:
    """A retrieved historical turn with its similarity score."""

    message: StoredMessage
    score: float


# Common short greetings, acknowledgments, and fillers that do not require context
_TRIVIAL_EXPRESSIONS: Set[str] = {
    "hi",
    "hi there",
    "hello",
    "hello there",
    "hey",
    "hey there",
    "howdy",
    "sup",
    "greetings",
    "good morning",
    "good evening",
    "good afternoon",
    "bye",
    "goodbye",
    "see ya",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "yes",
    "no",
    "cool",
    "great",
    "sure",
    "yep",
    "nope",
    "k",
    "whats up",
    "what's up",
}

_WORD_BOUNDARY_RE = re.compile(r"^\W+|\W+$")


class ContextRetriever:
    """
    Orchestrates selective context retrieval over ConversationStore.

    Parameters
    ----------
    conversation_store:
        The persistent conversation store backed by NeuroLinkDB.
    min_threshold:
        Minimum cosine similarity score (in [-1.0, 1.0]) required to inject a memory.
    max_items:
        Maximum number of past messages to retrieve.
    max_context_chars:
        Hard limit on the total character length of the retrieved context block.
    enabled:
        Master toggle for retrieval.
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        *,
        min_threshold: float = 0.25,
        max_items: int = 3,
        max_context_chars: int = 1200,
        enabled: bool = True,
    ) -> None:
        self._store = conversation_store
        self.min_threshold = min_threshold
        self.max_items = max_items
        self.max_context_chars = max_context_chars
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Intent / Heuristics
    # ------------------------------------------------------------------

    def should_retrieve(self, query: str) -> bool:
        """
        Decide whether context retrieval is appropriate for the given query.

        Returns False for short greetings, trivial acknowledgments, or empty queries.
        Returns True for substantive questions and informational queries.
        """
        if not self.enabled:
            return False

        cleaned = _WORD_BOUNDARY_RE.sub("", query.strip().lower())
        if not cleaned or len(cleaned) <= 2:
            return False

        if cleaned in _TRIVIAL_EXPRESSIONS:
            return False

        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        exclude_message_ids: Optional[Set[str]] = None,
        target_conversation_id: Optional[str] = None,
    ) -> List[RetrievedItem]:
        """
        Search and return semantically relevant historical messages.

        Parameters
        ----------
        query:
            The current user prompt.
        exclude_message_ids:
            Message IDs already present in the active turn window to prevent duplicate injection.
        target_conversation_id:
            Optional filter to restrict search to a single thread; if None, searches all threads.

        Returns
        -------
        List[RetrievedItem]
            Top relevant items that satisfy similarity threshold and character budget.
        """
        if not self.should_retrieve(query):
            return []

        if not self._store.vector_store or not self._store.embedding_provider:
            return []

        excluded = exclude_message_ids or set()

        raw_results = self._store.search_similar_messages(
            query,
            conversation_id=target_conversation_id,
            top_k=self.max_items * 2,  # Fetch extra candidates to account for exclusions
            threshold=self.min_threshold,
        )

        selected: List[RetrievedItem] = []
        accumulated_chars = 0

        for msg, score in raw_results:
            if msg.message_id in excluded:
                continue

            entry_len = len(msg.content) + len(msg.role) + 20
            if accumulated_chars + entry_len > self.max_context_chars and selected:
                break

            selected.append(RetrievedItem(message=msg, score=score))
            accumulated_chars += entry_len

            if len(selected) >= self.max_items:
                break

        return selected

    # ------------------------------------------------------------------
    # Prompt Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_context_prompt(items: List[RetrievedItem]) -> str:
        """
        Format retrieved items into a concise markdown section for the LLM prompt.
        """
        if not items:
            return ""

        lines = [
            "Relevant historical context retrieved from past conversations:",
            "---",
        ]
        for item in items:
            role = item.message.role
            content = item.message.content.strip()
            score_pct = int(round(item.score * 100))
            lines.append(f"• [{role} | relevance: {score_pct}%]: {content}")
        lines.append("---")
        return "\n".join(lines)
