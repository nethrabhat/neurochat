"""
tests/test_chat_service.py
──────────────────────────
Unit tests for services/chat_service.py.

A mock provider replaces Groq so tests run offline, without any API key,
and without installing the groq SDK.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from neurochat.llm.base import BaseLLMProvider, ChatResponse, Message
from neurochat.services.chat_service import ChatService


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class EchoProvider(BaseLLMProvider):
    """Returns a fixed reply for every call — useful for deterministic tests."""

    def __init__(self, reply: str = "echo reply") -> None:
        self._reply = reply
        self.calls: list[list[Message]] = []

    @property
    def provider_name(self) -> str:
        return "Echo"

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        return ChatResponse(content=self._reply, model="echo-v1", raw=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatServiceSend:
    def test_returns_provider_reply(self) -> None:
        provider = EchoProvider("Hello from mock!")
        service = ChatService(provider)
        response = service.send("Hi")
        assert response.content == "Hello from mock!"

    def test_raises_on_empty_message(self) -> None:
        service = ChatService(EchoProvider())
        with pytest.raises(ValueError, match="empty"):
            service.send("   ")

    def test_history_grows_after_send(self) -> None:
        service = ChatService(EchoProvider())
        service.send("First")
        service.send("Second")
        # 2 user + 2 assistant messages
        assert len(service.history) == 4

    def test_history_roles_are_correct(self) -> None:
        service = ChatService(EchoProvider("ok"))
        service.send("Question")
        roles = [m.role for m in service.history]
        assert roles == ["user", "assistant"]

    def test_history_content_matches_input(self) -> None:
        service = ChatService(EchoProvider("answer"))
        service.send("What is 2+2?")
        assert service.history[0].content == "What is 2+2?"
        assert service.history[1].content == "answer"


class TestChatServiceReset:
    def test_reset_clears_history(self) -> None:
        service = ChatService(EchoProvider())
        service.send("hello")
        service.reset()
        assert service.history == []

    def test_send_after_reset_works(self) -> None:
        service = ChatService(EchoProvider("fresh"))
        service.send("before")
        service.reset()
        response = service.send("after")
        assert response.content == "fresh"
        assert len(service.history) == 2


class TestChatServiceContextBuilding:
    def test_system_prompt_is_first_message(self) -> None:
        custom_prompt = "You are a pirate."
        provider = EchoProvider()
        service = ChatService(provider, system_prompt=custom_prompt)
        service.send("Ahoy!")
        first_msg = provider.calls[-1][0]
        assert first_msg.role == "system"
        assert first_msg.content == custom_prompt

    def test_context_trimmed_to_max_turns(self) -> None:
        provider = EchoProvider()
        # max_history_turns=2 → only 4 messages (2 user + 2 assistant) kept
        service = ChatService(provider, max_history_turns=2)
        for i in range(5):
            service.send(f"msg {i}")
        # Full history has 10 messages, but context window trims to 4+system
        last_call = provider.calls[-1]
        # 1 system + 4 history messages
        assert len(last_call) == 5

    def test_history_property_is_a_copy(self) -> None:
        """Mutating the returned list must not affect internal state."""
        service = ChatService(EchoProvider())
        service.send("hello")
        history_copy = service.history
        history_copy.clear()
        assert len(service.history) == 2


class TestProviderInterface:
    def test_echo_provider_satisfies_interface(self) -> None:
        """EchoProvider must be a valid BaseLLMProvider subclass."""
        assert isinstance(EchoProvider(), BaseLLMProvider)

    def test_chat_response_fields(self) -> None:
        provider = EchoProvider("hi")
        resp = provider.chat([Message(role="user", content="hello")])
        assert resp.content == "hi"
        assert resp.model == "echo-v1"
        assert resp.raw is None
