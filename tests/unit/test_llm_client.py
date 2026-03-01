"""Unit tests for the LLM client: JSON extraction, rate limiting, and call_llm error scenarios."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from prompter.config import Settings
from prompter.llm.client import (
    SchemaValidationError,
    _extract_json,
    _enforce_rate_limit,
    call_llm,
)


# ── Test model ────────────────────────────────────────────────────────


class SimpleModel(BaseModel):
    name: str
    value: int


# ── Helpers ───────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    defaults = {
        "groq_api_key": "test-key",
        "groq_model": "llama-3.3-70b-versatile",
        "max_retries": 3,
        "retry_base_delay": 0.01,
        "schema_retry_limit": 2,
        "rate_limit_tier": "paid",
        "free_tier_request_delay": 0.01,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _mock_response(content: str) -> MagicMock:
    """Create a mock LLM response with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _mock_client(*responses):
    """Create a mock ChatGroq client returning preset responses in order.

    Each response can be a string (returned as .content), a dict (JSON-dumped),
    or an Exception (raised on invoke).
    """
    client = MagicMock()
    side_effects = []
    for r in responses:
        if isinstance(r, Exception):
            side_effects.append(r)
        elif isinstance(r, dict):
            side_effects.append(_mock_response(json.dumps(r)))
        else:
            side_effects.append(_mock_response(str(r)))
    client.invoke.side_effect = side_effects
    return client


# ── JSON Extraction Tests ─────────────────────────────────────────────


class TestExtractJson:
    """Tests for _extract_json() helper."""

    def test_plain_json_passed_through(self):
        """Plain JSON string returned as-is after stripping."""
        raw = '{"name": "test", "value": 42}'
        assert _extract_json(raw) == raw

    def test_json_code_block_extracted(self):
        """JSON inside ```json code block is extracted."""
        raw = '```json\n{"name": "test", "value": 42}\n```'
        result = _extract_json(raw)
        assert json.loads(result) == {"name": "test", "value": 42}

    def test_generic_code_block_extracted(self):
        """JSON inside ``` code block (no language) is extracted."""
        raw = '```\n{"name": "test", "value": 42}\n```'
        result = _extract_json(raw)
        assert json.loads(result) == {"name": "test", "value": 42}

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        raw = '  \n  {"name": "test", "value": 1}  \n  '
        result = _extract_json(raw)
        assert json.loads(result) == {"name": "test", "value": 1}

    def test_multiline_json_between_fences(self):
        """Multi-line JSON between fences is extracted correctly."""
        raw = '```json\n{\n  "name": "test",\n  "value": 99\n}\n```'
        result = _extract_json(raw)
        assert json.loads(result) == {"name": "test", "value": 99}


# ── Rate Limit Tests ──────────────────────────────────────────────────


class TestEnforceRateLimit:
    """Tests for _enforce_rate_limit()."""

    @patch("prompter.llm.client.time")
    def test_free_tier_waits_between_requests(self, mock_time):
        """Free tier enforces minimum delay via time.sleep."""
        import prompter.llm.client as client_module

        original = client_module._last_request_time
        try:
            client_module._last_request_time = 100.0
            mock_time.time.return_value = 101.0  # 1s elapsed, need 4s more
            settings = _make_settings(rate_limit_tier="free", free_tier_request_delay=5.0)
            _enforce_rate_limit(settings)
            mock_time.sleep.assert_called_once()
            wait_arg = mock_time.sleep.call_args[0][0]
            assert 3.5 < wait_arg < 4.5
        finally:
            client_module._last_request_time = original

    @patch("prompter.llm.client.time")
    def test_paid_tier_skips_wait(self, mock_time):
        """Paid tier never sleeps."""
        mock_time.time.return_value = 1000.0
        settings = _make_settings(rate_limit_tier="paid")
        _enforce_rate_limit(settings)
        mock_time.sleep.assert_not_called()


# ── call_llm Tests ────────────────────────────────────────────────────


class TestCallLlm:
    """Tests for call_llm() with injected mock client."""

    def test_valid_response_returns_parsed_model(self):
        """Valid JSON response returns a correctly parsed Pydantic model."""
        client = _mock_client({"name": "test", "value": 42})
        result = call_llm(
            system_prompt="test system",
            user_message="test user",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert isinstance(result, SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_schema_appended_to_system_prompt(self):
        """The system prompt passed to invoke contains the JSON schema."""
        client = _mock_client({"name": "a", "value": 1})
        call_llm(
            system_prompt="You are helpful.",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        # Check the system message content
        call_args = client.invoke.call_args
        messages = call_args[0][0]
        system_content = messages[0].content
        assert "JSON" in system_content
        assert "name" in system_content
        assert "value" in system_content

    def test_injected_client_used_directly(self):
        """When client is provided, it is used directly (no new ChatGroq created)."""
        client = _mock_client({"name": "a", "value": 1})
        call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert client.invoke.called

    def test_malformed_json_triggers_self_healing(self):
        """Malformed JSON on first attempt triggers correction; valid on second succeeds."""
        client = _mock_client(
            "this is not json at all",                    # 1st: malformed
            {"name": "recovered", "value": 10},           # 2nd: valid
        )
        result = call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert result.name == "recovered"
        assert client.invoke.call_count == 2

    def test_pydantic_failure_triggers_correction(self):
        """Valid JSON with wrong fields triggers schema correction retry."""
        client = _mock_client(
            {"wrong_field": "bad"},                       # 1st: valid JSON, bad schema
            {"name": "fixed", "value": 99},               # 2nd: correct
        )
        result = call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert result.name == "fixed"
        assert result.value == 99

    def test_schema_exhaustion_raises_schema_validation_error(self):
        """After all schema retry attempts fail, SchemaValidationError is raised."""
        # schema_retry_limit=2 means 3 total attempts (initial + 2 retries)
        client = _mock_client(
            "bad1", "bad2", "bad3",
        )
        with pytest.raises(SchemaValidationError):
            call_llm(
                system_prompt="test",
                user_message="test",
                response_model=SimpleModel,
                settings=_make_settings(schema_retry_limit=2),
                client=client,
            )

    def test_http_error_retried_by_tenacity(self):
        """HTTP error on first attempt is retried by tenacity; second attempt succeeds."""
        client = _mock_client(
            ConnectionError("connection refused"),        # 1st: HTTP fail
            {"name": "ok", "value": 1},                   # 2nd: success
        )
        result = call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(max_retries=3),
            client=client,
        )
        assert result.name == "ok"

    def test_all_http_retries_exhausted_raises(self):
        """When all HTTP retries fail, the exception propagates."""
        client = _mock_client(
            ConnectionError("fail 1"),
            ConnectionError("fail 2"),
            ConnectionError("fail 3"),
        )
        with pytest.raises(ConnectionError):
            call_llm(
                system_prompt="test",
                user_message="test",
                response_model=SimpleModel,
                settings=_make_settings(max_retries=3),
                client=client,
            )

    def test_correction_message_contains_validation_error(self):
        """The correction message sent to LLM includes the validation error text."""
        client = _mock_client(
            {"wrong": "data"},                            # 1st: bad schema
            {"name": "fix", "value": 1},                  # 2nd: fixed
        )
        call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        # The second invocation should include correction text
        second_call = client.invoke.call_args_list[1]
        messages = second_call[0][0]
        # Correction is the last HumanMessage
        correction_content = messages[-1].content
        assert "validation error" in correction_content.lower() or "field required" in correction_content.lower()

    @patch("prompter.llm.client._enforce_rate_limit")
    def test_rate_limit_called_before_invoke(self, mock_rate_limit):
        """Rate limit enforcement is called before each LLM invocation."""
        client = _mock_client({"name": "a", "value": 1})
        call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert mock_rate_limit.called

    def test_response_with_markdown_code_block_parsed(self):
        """Response wrapped in ```json code block is correctly extracted and parsed."""
        client = _mock_client('```json\n{"name": "block", "value": 7}\n```')
        result = call_llm(
            system_prompt="test",
            user_message="test",
            response_model=SimpleModel,
            settings=_make_settings(),
            client=client,
        )
        assert result.name == "block"
        assert result.value == 7
