"""
llm/base.py
───────────
Abstract base class that every LLM provider adapter must implement.

Keeping provider-specific code out of this file means we can swap Groq
for OpenAI or a local model without touching the rest of the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    """A single turn in a conversation."""

    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Structured response returned by any LLM provider."""

    content: str
    model: str
    #: Raw provider-specific payload, kept for debugging.  May be None when
    #: the provider is mocked in tests.
    raw: dict | None = field(default=None, repr=False)
    #: Context memories retrieved from NeuroLinkDB, if any.
    retrieved_context: list | None = field(default=None)


class BaseLLMProvider(ABC):
    """Interface all LLM provider adapters must satisfy."""

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """
        Send *messages* to the model and return a structured response.

        Parameters
        ----------
        messages:
            Ordered list of conversation turns, including any system prompt.
        temperature:
            Sampling temperature (provider-specific behaviour).
        max_tokens:
            Upper bound on tokens in the generated reply.

        Returns
        -------
        ChatResponse
            Contains at least the reply text and the model identifier used.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"Groq"``."""
