"""
tests/test_context_retrieval.py
───────────────────────────────
Phase 5 test suite: Context-Aware Retrieval and Prompt Injection.

Covers:
  - Intent detection / should_retrieve heuristics (greetings vs substantive queries)
  - Selective retrieval: threshold filtering and relevance ranking
  - Bounded context: max_items and max_context_chars enforcement
  - De-duplication against current session context
  - Context prompt formatting
  - End-to-end ChatService cross-conversation memory retrieval with MockProvider
  - Toggling retrieval on and off

All tests run fully offline without external networks, APIs, or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from neurochat.neurolinkdb.database import NeuroLinkDB
from neurochat.embeddings.base import MockEmbeddingProvider
from neurochat.storage.vector_store import VectorStore
from neurochat.storage.conversation_store import ConversationStore, StoredMessage
from neurochat.services.retrieval_service import ContextRetriever, RetrievedItem
from neurochat.services.chat_service import ChatService
from neurochat.llm.base import BaseLLMProvider, ChatResponse, Message


# ─────────────────────────────────────────────────────────────────────────────
# Mock Provider capturing received prompts
# ─────────────────────────────────────────────────────────────────────────────

class ContextCapturingProvider(BaseLLMProvider):
    def __init__(self, reply: str = "mocked response") -> None:
        self.reply = reply
        self.received_prompts: list[list[Message]] = []

    @property
    def provider_name(self) -> str:
        return "CapturingMock"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        self.received_prompts.append(list(messages))
        return ChatResponse(content=self.reply, model="mock-model", raw=None)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    return tmp_path / "retrieval_test_db"


@pytest.fixture
def db(tmp_db_dir: Path) -> NeuroLinkDB:
    return NeuroLinkDB(tmp_db_dir)


@pytest.fixture
def embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=16, model_name="mock-embed-v1")


@pytest.fixture
def stores(
    db: NeuroLinkDB, embedder: MockEmbeddingProvider
) -> tuple[ConversationStore, VectorStore]:
    v_store = VectorStore(
        db,
        collection_name="vectors",
        dimension=embedder.dimension,
        model_name=embedder.model_name,
    )
    c_store = ConversationStore(
        db,
        vector_store=v_store,
        embedding_provider=embedder,
    )
    return c_store, v_store


@pytest.fixture
def retriever(stores: tuple[ConversationStore, VectorStore]) -> ContextRetriever:
    c_store, _ = stores
    return ContextRetriever(c_store, min_threshold=0.25, max_items=3, max_context_chars=500)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent Detection / Heuristics
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDetection:
    @pytest.mark.parametrize(
        "greeting",
        [
            "hi",
            "Hi!",
            "hello",
            "Hey there",
            "good morning",
            "thanks",
            "thank you!",
            "ok",
            "k",
            "yes",
            "sure",
            "bye",
        ],
    )
    def test_greetings_skip_retrieval(
        self, retriever: ContextRetriever, greeting: str
    ) -> None:
        assert retriever.should_retrieve(greeting) is False

    @pytest.mark.parametrize(
        "query",
        [
            "What did we discuss about database storage engines?",
            "Remember my favorite programming language?",
            "Can you explain how B-trees work in NeuroLinkDB?",
            "What was the name of my pet dog mentioned earlier?",
            "What is the capital of France?",
        ],
    )
    def test_substantive_queries_trigger_retrieval(
        self, retriever: ContextRetriever, query: str
    ) -> None:
        assert retriever.should_retrieve(query) is True

    def test_disabled_retriever_always_returns_false(
        self, retriever: ContextRetriever
    ) -> None:
        retriever.enabled = False
        assert (
            retriever.should_retrieve("What did we discuss about databases?")
            is False
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context Retrieval & Filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestContextRetrieval:
    def test_retrieve_relevant_memory(
        self, stores: tuple[ConversationStore, VectorStore], retriever: ContextRetriever
    ) -> None:
        c_store, _ = stores
        # Seed memory
        c_store.add_message(
            "thread-1",
            "user",
            "My secret project code name is Project Chimera.",
        )
        c_store.add_message(
            "thread-1",
            "assistant",
            "Understood, I will remember Project Chimera.",
        )
        c_store.add_message(
            "thread-2",
            "user",
            "I enjoy baking sourdough bread on weekends.",
        )

        # Query relevant to Project Chimera
        results = retriever.retrieve("What is my secret project code name?")
        assert len(results) > 0
        top_match = results[0]
        assert "Project Chimera" in top_match.message.content
        assert top_match.score >= retriever.min_threshold

    def test_threshold_filters_out_irrelevant_memories(
        self, stores: tuple[ConversationStore, VectorStore], retriever: ContextRetriever
    ) -> None:
        c_store, _ = stores
        c_store.add_message("thread-1", "user", "Quantum electrodynamics equations.")

        # Require a strict threshold that unrelated topics will not meet
        retriever.min_threshold = 0.95
        results = retriever.retrieve("How do I make chocolate chip cookies?")
        assert len(results) == 0

    def test_deduplication_against_excluded_ids(
        self, stores: tuple[ConversationStore, VectorStore], retriever: ContextRetriever
    ) -> None:
        c_store, _ = stores
        m1 = c_store.add_message("thread-1", "user", "Important memory detail.")

        # Without exclusion
        assert len(retriever.retrieve("Important memory detail")) > 0

        # With exclusion of m1
        results = retriever.retrieve(
            "Important memory detail",
            exclude_message_ids={m1.message_id},
        )
        assert not any(r.message.message_id == m1.message_id for r in results)

    def test_character_budget_bounding(
        self, stores: tuple[ConversationStore, VectorStore], retriever: ContextRetriever
    ) -> None:
        c_store, _ = stores
        retriever.max_context_chars = 150

        # Add multiple long messages
        for i in range(5):
            c_store.add_message(
                "thread-1",
                "user",
                f"Long message index {i} detailing architectural specs " * 3,
            )

        results = retriever.retrieve("architectural specs detailing")
        # Ensure we didn't greedily take all 5
        formatted = retriever.format_context_prompt(results)
        # Bounded within approximate character budget
        assert len(formatted) <= 350


# ─────────────────────────────────────────────────────────────────────────────
# 3. Context Prompt Formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptFormatting:
    def test_format_empty_returns_empty_string(self) -> None:
        assert ContextRetriever.format_context_prompt([]) == ""

    def test_format_with_items(self) -> None:
        msg = StoredMessage(
            conversation_id="c1",
            message_id="m1",
            role="user",
            content="Alice likes green apples.",
            timestamp=100.0,
        )
        item = RetrievedItem(message=msg, score=0.88)
        text = ContextRetriever.format_context_prompt([item])

        assert "Relevant historical context retrieved" in text
        assert "[user | relevance: 88%]: Alice likes green apples." in text
        assert "---" in text


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End ChatService Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestChatServiceRetrievalIntegration:
    def test_cross_conversation_memory_injection(
        self, stores: tuple[ConversationStore, VectorStore]
    ) -> None:
        c_store, _ = stores
        provider = ContextCapturingProvider("You like Rust because of memory safety.")

        # Session 1: User establishes a fact in thread-A
        service1 = ChatService(
            provider,
            conversation_store=c_store,
            conversation_id="thread-A",
        )
        service1.send("My favorite language is Rust because of memory safety.")

        # Session 2: User starts thread-B and asks about it
        service2 = ChatService(
            provider,
            conversation_store=c_store,
            conversation_id="thread-B",
        )
        response = service2.send("What programming language do I like and why?")

        # 1. Retrieved context should be attached to the response
        assert response.retrieved_context is not None
        assert len(response.retrieved_context) > 0
        retrieved_content = [
            item.message.content for item in response.retrieved_context
        ]
        assert any("Rust because of memory safety" in c for c in retrieved_content)

        # 2. Prompt sent to the LLM must contain the injected context block in system prompt
        last_prompt_sent = provider.received_prompts[-1]
        system_message = last_prompt_sent[0]
        assert system_message.role == "system"
        assert "Relevant historical context retrieved" in system_message.content
        assert "Rust because of memory safety" in system_message.content

    def test_greetings_do_not_inject_context(
        self, stores: tuple[ConversationStore, VectorStore]
    ) -> None:
        c_store, _ = stores
        provider = ContextCapturingProvider("Hello! How can I help?")

        # Seed database
        c_store.add_message("thread-A", "user", "My secret is 42.")

        service = ChatService(
            provider,
            conversation_store=c_store,
            conversation_id="thread-B",
        )
        response = service.send("Hello there!")

        # Greeting should not trigger context retrieval
        assert response.retrieved_context == []
        last_prompt = provider.received_prompts[-1]
        system_msg = last_prompt[0]
        assert "Relevant historical context retrieved" not in system_msg.content

