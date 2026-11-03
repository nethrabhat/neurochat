"""
services/chat_service.py
────────────────────────
Orchestrates one conversational turn:

  1. Optionally prepend a system prompt.
  2. Append the user message.
  3. Delegate to the configured LLM provider.
  4. Return the assistant reply.
  5. Persist turns to NeuroLinkDB when a ConversationStore is attached.

This service holds no provider-specific logic; it works with any object
that satisfies :class:`~neurochat.llm.base.BaseLLMProvider`.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Set

from neurochat.llm.base import BaseLLMProvider, ChatResponse, Message
from neurochat.storage.conversation_store import ConversationStore, StoredMessage
from neurochat.services.retrieval_service import ContextRetriever, RetrievedItem

_DEFAULT_SYSTEM_PROMPT = (
    "You are NeuroChat, a helpful and concise AI assistant. "
    "Answer clearly and honestly. "
    "If you do not know something, say so."
)


class ChatService:
    """High-level service that manages a single conversation session."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 20,
        conversation_store: Optional[ConversationStore] = None,
        conversation_id: Optional[str] = None,
        retriever: Optional[ContextRetriever] = None,
    ) -> None:
        """
        Parameters
        ----------
        provider:
            Any concrete :class:`BaseLLMProvider` implementation.
        system_prompt:
            Injected as the first message in every API call.
        max_history_turns:
            Number of most-recent *user+assistant* pairs kept in context.
            Each pair counts as 2 messages, so the window is
            ``max_history_turns * 2`` messages plus the system prompt.
        conversation_store:
            Optional :class:`ConversationStore` backed by NeuroLinkDB.
            When provided, messages are loaded from and saved to disk.
        conversation_id:
            Optional conversation ID to resume. If omitted and a store is
            provided, a new UUID4 string is generated.
        retriever:
            Optional :class:`ContextRetriever` for semantic cross-conversation
            memory retrieval. Auto-created if conversation_store has vector components.
        """
        self._provider = provider
        self._system_prompt = system_prompt
        self._max_history_turns = max_history_turns
        self._conversation_store = conversation_store
        self._conversation_id = (
            conversation_id or (str(uuid.uuid4()) if conversation_store else None)
        )
        self._history: List[Message] = []

        # Auto-instantiate retriever if components are available
        if retriever is not None:
            self._retriever = retriever
        elif (
            self._conversation_store
            and self._conversation_store.vector_store
            and self._conversation_store.embedding_provider
        ):
            self._retriever = ContextRetriever(self._conversation_store)
        else:
            self._retriever = None

        if self._conversation_store and self._conversation_id:
            self._load_persisted_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, user_text: str) -> ChatResponse:
        """
        Process one user turn and return the assistant response.

        The history is updated with both the user message and the reply,
        and turns are persisted to NeuroLinkDB if a store is configured.
        If context retrieval is active, relevant cross-conversation memories
        are retrieved and injected into the system prompt.

        Parameters
        ----------
        user_text:
            Raw text typed by the user.

        Returns
        -------
        ChatResponse
            The assistant reply plus provider metadata and retrieved context.
        """
        if not user_text.strip():
            raise ValueError("User message must not be empty.")

        user_msg = Message(role="user", content=user_text.strip())
        self._history.append(user_msg)

        user_stored: Optional[StoredMessage] = None
        if self._conversation_store and self._conversation_id:
            user_stored = self._conversation_store.add_message(
                conversation_id=self._conversation_id,
                role="user",
                content=user_msg.content,
            )

        # Context-aware retrieval
        retrieved_items: Optional[List[RetrievedItem]] = None
        if self._retriever:
            exclude_ids: Set[str] = set()
            if user_stored:
                exclude_ids.add(user_stored.message_id)

            # Also exclude all messages already in the active in-memory history
            if self._conversation_store and self._conversation_id:
                for past_msg in self._conversation_store.get_messages(self._conversation_id):
                    exclude_ids.add(past_msg.message_id)

            retrieved_items = self._retriever.retrieve(
                user_text,
                exclude_message_ids=exclude_ids,
            )

        messages = self._build_context(retrieved_items=retrieved_items)
        response = self._provider.chat(messages)
        response.retrieved_context = retrieved_items

        assistant_msg = Message(role="assistant", content=response.content)
        self._history.append(assistant_msg)

        if self._conversation_store and self._conversation_id:
            self._conversation_store.add_message(
                conversation_id=self._conversation_id,
                role="assistant",
                content=assistant_msg.content,
            )

        return response

    def reset(self, new_conversation_id: Optional[str] = None) -> None:
        """
        Clear in-memory conversation history to start a fresh turn.

        If a store is configured, a new conversation_id is assigned so that
        subsequent messages are stored in a new conversation thread.
        """
        self._history.clear()
        if self._conversation_store:
            self._conversation_id = new_conversation_id or str(uuid.uuid4())

    @property
    def history(self) -> List[Message]:
        """Read-only view of the current conversation history."""
        return list(self._history)

    @property
    def conversation_id(self) -> Optional[str]:
        """Identifier for the current active conversation thread."""
        return self._conversation_id

    @property
    def conversation_store(self) -> Optional[ConversationStore]:
        """Underlying conversation store, if configured."""
        return self._conversation_store

    @property
    def retriever(self) -> Optional[ContextRetriever]:
        """Active context retriever, if configured."""
        return self._retriever

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_persisted_history(self) -> None:
        """Load past messages from the store into in-memory history."""
        if not self._conversation_store or not self._conversation_id:
            return

        stored = self._conversation_store.get_messages(self._conversation_id)
        for msg in stored:
            if msg.role in ("user", "assistant"):
                self._history.append(msg.to_message())

    def _build_context(
        self, retrieved_items: Optional[List[RetrievedItem]] = None
    ) -> List[Message]:
        """
        Build the message list sent to the provider.

        Layout:
          [system prompt + optional retrieved context] + [trimmed history]

        The most-recent messages are kept when trimming to stay within
        ``max_history_turns``.
        """
        system_content = self._system_prompt
        if retrieved_items and self._retriever:
            context_block = self._retriever.format_context_prompt(retrieved_items)
            if context_block:
                system_content = f"{self._system_prompt}\n\n{context_block}"

        system = Message(role="system", content=system_content)

        # Each turn = 1 user + 1 assistant message. Keep the last N turns.
        max_messages = self._max_history_turns * 2
        trimmed_history = self._history[-max_messages:]

        return [system] + trimmed_history
