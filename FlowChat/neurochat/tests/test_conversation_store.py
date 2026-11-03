"""
tests/test_conversation_store.py
─────────────────────────────────
Phase 3 test suite for Conversation persistence with NeuroLinkDB.

Covers:
  - StoredMessage data class and round-trip conversions
  - ConversationStore insert and chronological retrieval
  - Isolation between multiple conversations
  - Persistence across database close and reopen
  - Listing conversations with metadata
  - Deleting conversations (tombstoning)
  - Integration of ChatService with ConversationStore
  - Restoring conversation history in a new ChatService instance

All tests run fully offline without network or external services.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.storage.conversation_store import ConversationStore, StoredMessage
from neurochat.services.chat_service import ChatService
from neurochat.llm.base import BaseLLMProvider, ChatResponse, Message


# ---------------------------------------------------------------------------
# Mock LLM provider for deterministic offline testing
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    def __init__(self, reply: str = "mock reply") -> None:
        self.reply = reply
        self.received_messages: list[list[Message]] = []

    @property
    def provider_name(self) -> str:
        return "Mock"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        self.received_messages.append(list(messages))
        return ChatResponse(content=self.reply, model="mock-model", raw=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    return tmp_path / "conv_db"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def store(db: NeuroLinkDB) -> ConversationStore:
    return ConversationStore(db)


# ---------------------------------------------------------------------------
# Tests: StoredMessage
# ---------------------------------------------------------------------------

class TestStoredMessage:
    def test_instantiation_defaults(self) -> None:
        msg = StoredMessage(
            conversation_id="c1",
            message_id="m1",
            role="user",
            content="Hello world",
            timestamp=1000.0,
        )
        assert msg.conversation_id == "c1"
        assert msg.message_id == "m1"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.timestamp == 1000.0
        assert msg.embedding_ref is None

    def test_dict_roundtrip(self) -> None:
        msg = StoredMessage(
            conversation_id="conv-123",
            message_id="msg-456",
            role="assistant",
            content="Persistent reply",
            timestamp=123456.78,
            embedding_ref="emb:vec:1",
        )
        d = msg.to_dict()
        msg2 = StoredMessage.from_dict(d)
        assert msg2 == msg

    def test_to_message_conversion(self) -> None:
        stored = StoredMessage(
            conversation_id="c",
            message_id="m",
            role="user",
            content="Hi",
            timestamp=1.0,
        )
        llm_msg = stored.to_message()
        assert isinstance(llm_msg, Message)
        assert llm_msg.role == "user"
        assert llm_msg.content == "Hi"


# ---------------------------------------------------------------------------
# Tests: ConversationStore operations
# ---------------------------------------------------------------------------

class TestConversationStore:
    def test_add_and_get_messages(self, store: ConversationStore) -> None:
        cid = "conv-1"
        m1 = store.add_message(cid, "user", "Hello", timestamp=100.0)
        m2 = store.add_message(cid, "assistant", "Hi there!", timestamp=101.0)

        messages = store.get_messages(cid)
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[0].role == "user"
        assert messages[1].content == "Hi there!"
        assert messages[1].role == "assistant"

    def test_message_sorting_by_timestamp(self, store: ConversationStore) -> None:
        cid = "conv-order"
        # Insert out of order
        store.add_message(cid, "user", "Second", timestamp=200.0)
        store.add_message(cid, "user", "First", timestamp=100.0)
        store.add_message(cid, "user", "Third", timestamp=300.0)

        messages = store.get_messages(cid)
        contents = [m.content for m in messages]
        assert contents == ["First", "Second", "Third"]

    def test_multiple_conversations_isolated(self, store: ConversationStore) -> None:
        store.add_message("conv-A", "user", "Msg in A")
        store.add_message("conv-B", "user", "Msg in B")

        msgs_a = store.get_messages("conv-A")
        msgs_b = store.get_messages("conv-B")

        assert len(msgs_a) == 1
        assert msgs_a[0].content == "Msg in A"
        assert len(msgs_b) == 1
        assert msgs_b[0].content == "Msg in B"

    def test_invalid_arguments_raise(self, store: ConversationStore) -> None:
        with pytest.raises(ValueError, match="conversation_id"):
            store.add_message("   ", "user", "text")

        with pytest.raises(ValueError, match="role"):
            store.add_message("conv", "   ", "text")

        with pytest.raises(ValueError, match="content"):
            store.add_message("conv", "user", "   ")

    def test_list_conversations_summary(self, store: ConversationStore) -> None:
        store.add_message("c1", "user", "Hello from c1", timestamp=10.0)
        store.add_message("c1", "assistant", "Reply in c1", timestamp=11.0)

        store.add_message("c2", "user", "Hello from c2", timestamp=20.0)

        convs = store.list_conversations()
        assert len(convs) == 2

        # Most recently updated conversation comes first
        assert convs[0]["conversation_id"] == "c2"
        assert convs[0]["message_count"] == 1
        assert convs[0]["last_message"] == "Hello from c2"

        assert convs[1]["conversation_id"] == "c1"
        assert convs[1]["message_count"] == 2
        assert convs[1]["last_message"] == "Reply in c1"

    def test_delete_conversation(self, store: ConversationStore) -> None:
        store.add_message("del-conv", "user", "Delete me")
        store.add_message("del-conv", "assistant", "I will be deleted")
        store.add_message("keep-conv", "user", "Keep me")

        count = store.delete_conversation("del-conv")
        assert count == 2

        assert store.get_messages("del-conv") == []
        assert len(store.get_messages("keep-conv")) == 1


# ---------------------------------------------------------------------------
# Tests: Persistence across restarts
# ---------------------------------------------------------------------------

class TestConversationPersistence:
    def test_messages_survive_reopen(self, tmp_db_dir: Path) -> None:
        # Session 1: write conversation
        db1 = NeuroLinkDB(tmp_db_dir)
        store1 = ConversationStore(db1)
        store1.add_message("persisted-conv", "user", "Remember this message")
        store1.add_message("persisted-conv", "assistant", "Yes, stored in NeuroLinkDB")
        db1.close()

        # Session 2: reopen database
        db2 = NeuroLinkDB(tmp_db_dir)
        store2 = ConversationStore(db2)
        recovered = store2.get_messages("persisted-conv")

        assert len(recovered) == 2
        assert recovered[0].content == "Remember this message"
        assert recovered[1].content == "Yes, stored in NeuroLinkDB"
        db2.close()


# ---------------------------------------------------------------------------
# Tests: ChatService persistence integration
# ---------------------------------------------------------------------------

class TestChatServicePersistence:
    def test_chat_service_persists_turns(self, store: ConversationStore) -> None:
        provider = MockProvider("Assistant answer")
        service = ChatService(
            provider,
            conversation_store=store,
            conversation_id="chat-1",
        )

        resp = service.send("User question")
        assert resp.content == "Assistant answer"

        # Check that both user and assistant turns are stored
        stored = store.get_messages("chat-1")
        assert len(stored) == 2
        assert stored[0].role == "user"
        assert stored[0].content == "User question"
        assert stored[1].role == "assistant"
        assert stored[1].content == "Assistant answer"

    def test_chat_service_restores_history_on_reconnect(
        self, tmp_db_dir: Path
    ) -> None:
        # First session
        db1 = NeuroLinkDB(tmp_db_dir)
        store1 = ConversationStore(db1)
        provider1 = MockProvider("First reply")
        service1 = ChatService(
            provider1,
            conversation_store=store1,
            conversation_id="session-xyz",
        )
        service1.send("First question")
        db1.close()

        # Second session: reconnect to same conversation_id
        db2 = NeuroLinkDB(tmp_db_dir)
        store2 = ConversationStore(db2)
        provider2 = MockProvider("Second reply")
        service2 = ChatService(
            provider2,
            conversation_store=store2,
            conversation_id="session-xyz",
        )

        # In-memory history should be restored immediately from the store
        assert len(service2.history) == 2
        assert service2.history[0].content == "First question"
        assert service2.history[1].content == "First reply"

        # Sending another turn appends to existing history
        service2.send("Second question")
        assert len(service2.history) == 4

        # Verify context sent to provider includes earlier history
        last_call_messages = provider2.received_messages[-1]
        # System prompt + 2 restored messages + 1 new user message = 4
        assert len(last_call_messages) == 4
        db2.close()

    def test_chat_service_reset_creates_new_conversation(
        self, store: ConversationStore
    ) -> None:
        provider = MockProvider()
        service = ChatService(provider, conversation_store=store)
        initial_cid = service.conversation_id
        assert initial_cid is not None

        service.send("Msg in conv 1")
        assert len(store.get_messages(initial_cid)) == 2

        service.reset()
        new_cid = service.conversation_id
        assert new_cid != initial_cid
        assert service.history == []

        service.send("Msg in conv 2")
        assert len(store.get_messages(new_cid)) == 2
        # Previous conversation is still safely preserved in DB
        assert len(store.get_messages(initial_cid)) == 2

