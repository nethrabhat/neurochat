"""
llm/groq_client.py
──────────────────
Groq LLM provider adapter.

⚠️  INTEGRATION NOTE — verification required during setup
─────────────────────────────────────────────────────────
This wrapper is written against the *expected* public API of the official
``groq`` Python SDK (https://pypi.org/project/groq/).  The SDK has NOT been
installed yet, so the import is guarded and the actual call is isolated in
``_call_groq_api``.  When the SDK is installed and a real API key is
available, run the project end-to-end and verify:

1.  ``from groq import Groq`` succeeds.
2.  ``client.chat.completions.create(model=..., messages=...)`` is the
    correct call signature for the installed version.
3.  The response object exposes
    ``response.choices[0].message.content`` and ``response.model``.

Adjust ``_call_groq_api`` if the SDK differs from these expectations.
"""

from __future__ import annotations

from typing import List

from neurochat.config.settings import Settings, MissingAPIKeyError
from neurochat.llm.base import BaseLLMProvider, ChatResponse, Message

# ---------------------------------------------------------------------------
# SDK import — kept isolated so tests can run without the package installed.
# ---------------------------------------------------------------------------
try:
    from groq import Groq as _GroqClient  # type: ignore[import]

    _SDK_AVAILABLE = True
except ImportError:
    _GroqClient = None  # type: ignore[assignment,misc]
    _SDK_AVAILABLE = False


class GroqProvider(BaseLLMProvider):
    """Adapter that speaks to Groq's chat-completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: object | None = None  # lazily initialised

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "Groq"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """
        Forward *messages* to Groq and return a :class:`ChatResponse`.

        Raises
        ------
        MissingAPIKeyError
            When ``GROQ_API_KEY`` is not configured.
        RuntimeError
            When the ``groq`` SDK is not installed.
        """
        api_key = self._settings.require_groq_api_key()

        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "The 'groq' package is not installed. "
                "Run: pip install groq  (after activating the virtual env)"
            )

        client = self._get_client(api_key)
        raw_messages = [m.to_dict() for m in messages]

        return self._call_groq_api(
            client=client,
            model=self._settings.model_name,
            messages=raw_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers — isolated to make future verification easier
    # ------------------------------------------------------------------

    def _get_client(self, api_key: str) -> object:
        """Return (and cache) the Groq SDK client."""
        if self._client is None:
            # ⚠️ Verify: Groq(api_key=...) is the correct constructor signature.
            self._client = _GroqClient(api_key=api_key)  # type: ignore[misc]
        return self._client

    @staticmethod
    def _call_groq_api(
        client: object,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        """
        Perform the actual SDK call.

        ⚠️ Verify this signature matches the installed groq SDK version:
            client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        """
        # We access the SDK through `object` type to avoid static-analysis
        # errors when the package is not installed.
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ⚠️ Verify: confirm response structure with the installed SDK.
        content: str = response.choices[0].message.content  # type: ignore[union-attr]
        used_model: str = getattr(response, "model", model)

        return ChatResponse(
            content=content,
            model=used_model,
            raw={"id": getattr(response, "id", None)},
        )
