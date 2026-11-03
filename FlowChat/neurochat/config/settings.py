"""
config/settings.py
──────────────────
Application-level configuration loaded from environment variables.
A .env file is NOT loaded automatically here; callers that want dotenv
support should load it before importing this module (e.g. via python-dotenv).
"""

from __future__ import annotations

import os


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is absent from the environment."""


class Settings:
    """Holds validated application configuration."""

    #: Default model used with Groq when MODEL_NAME is not set.
    DEFAULT_MODEL: str = "openai/gpt-oss-120b"

    #: Default directory where NeuroLinkDB persists collections.
    DEFAULT_DB_DIR: str = "./data/neurolinkdb"

    def __init__(self) -> None:
        self.groq_api_key: str | None = os.getenv("GROQ_API_KEY")
        self.model_name: str = os.getenv("MODEL_NAME", self.DEFAULT_MODEL)
        self.app_title: str = os.getenv("APP_TITLE", "NeuroChat")
        self.db_dir: str = os.getenv("DB_DIR", self.DEFAULT_DB_DIR)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def require_groq_api_key(self) -> str:
        """Return the Groq API key or raise :class:`MissingAPIKeyError`."""
        if not self.groq_api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your key, "
                "then restart the application."
            )
        return self.groq_api_key

    def is_groq_configured(self) -> bool:
        """Return True when a non-empty GROQ_API_KEY is present."""
        return bool(self.groq_api_key)

    def __repr__(self) -> str:  # pragma: no cover
        key_preview = "***" if self.groq_api_key else "<not set>"
        return (
            f"Settings(groq_api_key={key_preview!r}, "
            f"model_name={self.model_name!r}, "
            f"app_title={self.app_title!r}, "
            f"db_dir={self.db_dir!r})"
        )


# Module-level singleton — import this where convenient.
settings = Settings()
