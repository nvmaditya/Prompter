"""Tests for CLI help text, error messages, and input validation polish."""

from unittest.mock import patch

from typer.testing import CliRunner

from prompter.cli import app


runner = CliRunner()


# ── Help and Version Tests ────────────────────────────────────────────


class TestCLIHelpAndVersion:
    """Tests for CLI help output and version flag."""

    def test_no_args_shows_help(self):
        """Running prompter with no args shows help/usage text."""
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True exits with code 0 or 2 depending on version
        assert result.exit_code in (0, 2)
        assert "generate" in result.output.lower() or "Usage" in result.output

    def test_version_flag(self):
        """--version prints the version string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_generate_help(self):
        """generate --help shows command options."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output
        assert "--resume" in result.output
        assert "--verbose" in result.output


# ── Error Message Tests ───────────────────────────────────────────────


class TestCLIErrorMessages:
    """Tests for user-facing error messages."""

    @patch("prompter.config.Settings.__init__")
    def test_missing_api_key_gives_clear_message(self, mock_settings_init):
        """Missing GROQ_API_KEY produces a clear error message with exit code 3."""
        mock_settings_init.side_effect = Exception("GROQ_API_KEY field required")
        result = runner.invoke(app, ["generate", "a reasonable length test idea"])
        assert result.exit_code == 3
        assert "GROQ_API_KEY" in result.output or "Configuration error" in result.output

    def test_too_short_input_shows_feedback(self):
        """Input shorter than 10 chars shows specific error feedback."""
        result = runner.invoke(app, ["generate", "hi"])
        assert result.exit_code == 1
        assert "too short" in result.output.lower()

    def test_too_long_input_shows_feedback(self):
        """Input longer than 10,000 chars shows specific error feedback."""
        long_input = "a" * 10_001
        result = runner.invoke(app, ["generate", long_input])
        assert result.exit_code == 1
        assert "too long" in result.output.lower()
