"""
tests/test_settings.py
──────────────────────
Unit tests for config/settings.py.

These tests manipulate environment variables directly and require no
internet access, no API keys, and no external packages beyond pytest.
"""

from __future__ import annotations

import os
import importlib
import pytest

from neurochat.config.settings import Settings, MissingAPIKeyError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**env_overrides: str | None) -> Settings:
    """Create a fresh Settings instance with controlled env vars."""
    # Temporarily patch os.environ for the duration of this call.
    original: dict[str, str] = {}
    keys_to_remove: list[str] = []

    env_keys = ("GROQ_API_KEY", "MODEL_NAME", "APP_TITLE", "DB_DIR")
    for key in env_keys:
        if key in os.environ:
            original[key] = os.environ.pop(key)

    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        return Settings()
    finally:
        # Restore environment
        for key in env_keys:
            os.environ.pop(key, None)
        for key, value in original.items():
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    def test_default_model_when_not_set(self) -> None:
        s = _make_settings(GROQ_API_KEY=None, MODEL_NAME=None)
        assert s.model_name == Settings.DEFAULT_MODEL

    def test_default_title_when_not_set(self) -> None:
        s = _make_settings(GROQ_API_KEY=None, APP_TITLE=None)
        assert s.app_title == "NeuroChat"

    def test_default_db_dir_when_not_set(self) -> None:
        s = _make_settings(GROQ_API_KEY=None, DB_DIR=None)
        assert s.db_dir == Settings.DEFAULT_DB_DIR

    def test_api_key_is_none_when_not_set(self) -> None:
        s = _make_settings(GROQ_API_KEY=None)
        assert s.groq_api_key is None


class TestSettingsWithValues:
    def test_api_key_read_from_env(self) -> None:
        s = _make_settings(GROQ_API_KEY="test-key-abc")
        assert s.groq_api_key == "test-key-abc"

    def test_model_name_read_from_env(self) -> None:
        s = _make_settings(MODEL_NAME="mixtral-8x7b-32768")
        assert s.model_name == "mixtral-8x7b-32768"

    def test_app_title_read_from_env(self) -> None:
        s = _make_settings(APP_TITLE="MyBot")
        assert s.app_title == "MyBot"

    def test_db_dir_read_from_env(self) -> None:
        s = _make_settings(DB_DIR="/custom/db/path")
        assert s.db_dir == "/custom/db/path"


class TestIsGroqConfigured:
    def test_returns_false_when_key_absent(self) -> None:
        s = _make_settings(GROQ_API_KEY=None)
        assert s.is_groq_configured() is False

    def test_returns_true_when_key_present(self) -> None:
        s = _make_settings(GROQ_API_KEY="gsk_fake")
        assert s.is_groq_configured() is True

    def test_returns_false_for_empty_string(self) -> None:
        s = _make_settings(GROQ_API_KEY="")
        assert s.is_groq_configured() is False


class TestRequireGroqApiKey:
    def test_raises_when_key_absent(self) -> None:
        s = _make_settings(GROQ_API_KEY=None)
        with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
            s.require_groq_api_key()

    def test_returns_key_when_present(self) -> None:
        s = _make_settings(GROQ_API_KEY="gsk_valid")
        assert s.require_groq_api_key() == "gsk_valid"
