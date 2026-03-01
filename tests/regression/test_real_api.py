"""Real API regression tests — runs prompter against the live Groq API.

These tests are marked @pytest.mark.slow and excluded from the default test run.
Run them explicitly with: pytest tests/regression/ -m slow -v
"""

import pytest
from typer.testing import CliRunner

from prompter.cli import app
from prompter.models.final_output import FinalOutputArtifact


@pytest.fixture(scope="module")
def real_api_output(tmp_path_factory):
    """Run the full pipeline once against the real Groq API.

    All tests in this module share this single pipeline execution
    to avoid redundant API calls and rate limit issues.
    """
    output_dir = tmp_path_factory.mktemp("real_api_output")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["generate", "a quizzing platform for medical students", "-o", str(output_dir)],
    )
    return {"result": result, "output_dir": output_dir}


@pytest.mark.slow
class TestRealAPIGeneration:
    """Smoke tests verifying the pipeline produces valid output with real API calls."""

    def test_generate_exits_zero(self, real_api_output):
        """Pipeline completes without error (exit code 0)."""
        result = real_api_output["result"]
        assert result.exit_code == 0, (
            f"Pipeline failed with exit code {result.exit_code}:\n{result.output}"
        )

    def test_json_valid_schema(self, real_api_output):
        """prompt_config.json exists and validates as FinalOutputArtifact."""
        output_dir = real_api_output["output_dir"]
        json_path = output_dir / "prompt_config.json"
        assert json_path.exists(), f"JSON output not found at {json_path}"

        content = json_path.read_text(encoding="utf-8")
        artifact = FinalOutputArtifact.model_validate_json(content)
        assert len(artifact.modules) >= 1
        assert artifact.project is not None

    def test_markdown_nonempty(self, real_api_output):
        """architecture_spec.md exists and has substantial content."""
        output_dir = real_api_output["output_dir"]
        md_path = output_dir / "architecture_spec.md"
        assert md_path.exists(), f"Markdown output not found at {md_path}"

        content = md_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"Markdown is too short ({len(content)} chars)"

    def test_scaffolding_structure(self, real_api_output):
        """scaffolding/ directory has the expected structure."""
        output_dir = real_api_output["output_dir"]
        scaffold_dir = output_dir / "scaffolding"
        assert scaffold_dir.is_dir(), f"Scaffolding dir not found at {scaffold_dir}"

        assert (scaffold_dir / "prompts").is_dir()
        assert (scaffold_dir / "agents").is_dir()
        assert (scaffold_dir / "README.md").exists()
        assert (scaffold_dir / "config.py").exists()
        assert (scaffold_dir / "main.py").exists()
