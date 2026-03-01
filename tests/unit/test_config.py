"""Unit tests for Settings configuration."""

import os

import pytest

from prompter.config import Settings


class TestSettings:
    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        settings = Settings(_env_file=None)
        assert settings.groq_api_key == "test-key-123"
        assert settings.groq_model == "llama-3.3-70b-versatile"

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        settings = Settings(_env_file=None)
        assert settings.quality_threshold == 7.0
        assert settings.max_iterations == 3
        assert settings.creative_temperature == 0.7
        assert settings.analytical_temperature == 0.3
        assert settings.packager_temperature == 0.5
        assert settings.max_retries == 3
        assert settings.schema_retry_limit == 2
        assert settings.rate_limit_tier == "free"
        assert settings.free_tier_request_delay == 5.0
        assert settings.verbose is False

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("PROMPTER_QUALITY_THRESHOLD", "8.5")
        monkeypatch.setenv("PROMPTER_MAX_ITERATIONS", "5")
        monkeypatch.setenv("PROMPTER_VERBOSE", "true")
        settings = Settings(_env_file=None)
        assert settings.quality_threshold == 8.5
        assert settings.max_iterations == 5
        assert settings.verbose is True

    def test_missing_api_key_fails(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        # Also remove any .env file influence
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_safe_dict_excludes_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "secret-key")
        settings = Settings(_env_file=None)
        safe = settings.safe_dict()
        assert "groq_api_key" not in safe
        assert "groq_model" in safe

    def test_rate_limit_tier_values(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("PROMPTER_RATE_LIMIT_TIER", "paid")
        settings = Settings(_env_file=None)
        assert settings.rate_limit_tier == "paid"
