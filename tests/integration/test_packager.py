"""Integration tests for the Packager agent and output writers."""

import json
import py_compile
from pathlib import Path
from unittest.mock import patch

import pytest

from prompter.agents.packager import NarrativeResponse, package
from prompter.config import Settings
from prompter.models.critic_feedback import (
    CategoryScore,
    CriticFeedback,
    Issue,
    IssueCategory,
    Severity,
)
from prompter.models.final_output import FinalOutputArtifact
from prompter.models.inter_agent_map import (
    HandoffCondition,
    InterAgentMap,
    SharedMemoryField,
    Trigger,
)
from prompter.models.module_map import (
    DomainClassification,
    InteractionType,
    Module,
    ModuleMap,
)
from prompter.models.prompt_artifact import (
    ContextSlot,
    EvalCriteria,
    PromptArtifact,
    TokenEstimate,
)
from prompter.output.json_writer import write_json
from prompter.output.markdown_writer import write_markdown
from prompter.output.scaffold_writer import write_scaffolding
from prompter.state import create_initial_state


def _make_settings(**overrides) -> Settings:
    defaults = {"groq_api_key": "test-key", "groq_model": "llama-3.3-70b-versatile"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_artifact(module_name: str = "Question Generator") -> PromptArtifact:
    return PromptArtifact(
        module_name=module_name,
        agent_role=f"{module_name} Specialist",
        primary_technique="chain_of_thought",
        secondary_technique="output_constraints",
        technique_rationale=f"CoT for {module_name}.",
        system_prompt=(
            f"You are a {module_name} Specialist.\n\n"
            f"## Constraints\n- NEVER fabricate information\n- ALWAYS cite sources\n\n"
            f"Input: {{{{input_data}}}}\nLevel: {{{{user_level}}}}\n\n"
            f"## Output Format\nRespond with valid JSON."
        ),
        context_slots=[
            ContextSlot(
                variable="input_data",
                description="Primary input",
                source="Previous module",
                injection_time="runtime",
                fallback="N/A",
                required=True,
            ),
            ContextSlot(
                variable="user_level",
                description="Learner proficiency",
                source="Tracker",
                injection_time="session_start",
                fallback="intermediate",
                required=False,
            ),
        ],
        token_estimate=TokenEstimate(
            system_tokens=300, expected_context_tokens=200,
            expected_output_tokens=200, total=700,
        ),
        triggers=["module_invoked"],
        outputs_to=["Quiz Session Manager"],
        eval_criteria=EvalCriteria(
            good_output_examples=["Good 1", "Good 2"],
            bad_output_examples=["Bad 1", "Bad 2"],
            automated_eval_suggestions=["Check JSON", "Verify refs", "Validate tokens"],
            human_review_criteria=["Domain accuracy"],
        ),
    )


def _make_module_map() -> ModuleMap:
    return ModuleMap(
        project_name="Medical Quiz Platform",
        domain_classification=DomainClassification(primary="Education"),
        interaction_model=InteractionType.adaptive,
        interaction_model_rationale="Adaptive difficulty.",
        modules=[
            Module(
                name="Question Generator",
                description="Generates quiz questions.",
                requires_ai=True,
                ai_justification="NLG.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.batch,
                data_inputs=["topic"],
                data_outputs=["question"],
                failure_modes=["Off-topic"],
            ),
            Module(
                name="Feedback Generator",
                description="Produces feedback.",
                requires_ai=True,
                ai_justification="NLG.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.conversational,
                data_inputs=["question"],
                data_outputs=["feedback_text"],
                failure_modes=["Generic"],
                depends_on=["Question Generator"],
            ),
        ],
        module_count=2,
        ai_module_count=2,
    )


def _make_inter_agent_map() -> InterAgentMap:
    return InterAgentMap(
        shared_memory_schema={
            "current_topic": SharedMemoryField(
                type="string",
                description="Active quiz topic",
                written_by=["Question Generator"],
                read_by=["Feedback Generator"],
                updated_on="question_generated",
                default="",
            ),
        },
        handoff_conditions=[
            HandoffCondition(
                from_agent="Question Generator",
                to_agent="Feedback Generator",
                condition="question_generated == true",
                data_passed={"question": "Generated question text"},
                format="json",
                fallback_if_incomplete="Use default question template",
            ),
        ],
        context_pollution_rules=[],
        trigger_map=[
            Trigger(
                event="quiz_started",
                activates=["Question Generator"],
                priority_order=["Question Generator"],
                execution="sequential",
                error_fallback="Retry with default topic",
            ),
        ],
    )


def _make_feedback(module_name: str, score: float = 8.2, passed: bool = True) -> CriticFeedback:
    issues = []
    if not passed:
        issues.append(Issue(
            category=IssueCategory.hallucination_risk,
            severity=Severity.high,
            location="system prompt",
            description="No grounding constraints.",
            suggestion="Add grounding rules.",
        ))
    return CriticFeedback(
        module_name=module_name,
        overall_score=score,
        passed=passed,
        category_scores={
            "ambiguity": CategoryScore(score=8.5),
            "hallucination_risk": CategoryScore(score=8.0),
            "missing_constraints": CategoryScore(score=8.0),
            "edge_cases": CategoryScore(score=8.0),
            "token_efficiency": CategoryScore(score=8.5),
        },
        issues=issues,
        iteration=1,
        summary=f"Assessment for {module_name}.",
    )


def _make_full_state() -> dict:
    """Create a fully-populated state for packager testing."""
    settings = _make_settings()
    state = create_initial_state(
        project_idea="a medical quizzing platform",
        config=settings.safe_dict(),
        run_id="test-run-001",
    )
    state["module_map"] = _make_module_map()
    state["prompt_artifacts"] = [
        _make_artifact("Question Generator"),
        _make_artifact("Feedback Generator"),
    ]
    state["inter_agent_map"] = _make_inter_agent_map()
    state["critic_feedback"] = [[
        _make_feedback("Question Generator"),
        _make_feedback("Feedback Generator"),
    ]]
    state["current_iteration"] = 1
    state["all_passed"] = True
    state["agent_durations"] = {"analyzer": 1.5, "architect": 3.2, "critic": 2.0}
    state["token_usage"] = {"analyzer": 500, "architect": 1200, "critic": 800}
    return state


# ── Packager Node Tests ──────────────────────────────────────────────


class TestPackagerNode:
    """Tests for the package() agent node with mocked LLM."""

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_produces_valid_final_output(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """Packager produces a valid FinalOutputArtifact."""
        mock_llm.return_value = NarrativeResponse(narrative="Executive summary text.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        state = _make_full_state()
        result = package(state, settings=_make_settings())

        assert result["final_output"] is not None
        assert isinstance(result["final_output"], FinalOutputArtifact)

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_correct_module_count(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """FinalOutputArtifact has the correct number of modules."""
        mock_llm.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        state = _make_full_state()
        result = package(state, settings=_make_settings())

        assert len(result["final_output"].modules) == 2

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_metadata_computed_correctly(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """PipelineMetadata has correct computed values."""
        mock_llm.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        state = _make_full_state()
        result = package(state, settings=_make_settings())

        meta = result["final_output"].pipeline_metadata
        assert meta.total_modules == 2
        assert meta.ai_modules == 2
        assert meta.total_estimated_tokens == 1400  # 700 * 2
        assert meta.average_quality_score == 8.2
        assert meta.critic_iterations_used == 1

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_telemetry_populated(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """Packager populates agent_durations and last_checkpoint."""
        mock_llm.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        state = _make_full_state()
        result = package(state, settings=_make_settings())

        assert "packager" in result["agent_durations"]
        assert result["agent_durations"]["packager"] >= 0
        assert result["last_checkpoint"] == "package"

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_calls_llm_for_narrative(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """Packager calls LLM exactly once for narrative generation."""
        mock_llm.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        state = _make_full_state()
        package(state, settings=_make_settings())

        assert mock_llm.call_count == 1
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("response_model") == NarrativeResponse

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    def test_raises_without_inter_agent_map(self, mock_llm, mock_json, mock_md, mock_scaffold):
        """Packager raises ValueError when inter_agent_map is None."""
        mock_llm.return_value = NarrativeResponse(narrative="Summary.")

        state = _make_full_state()
        state["inter_agent_map"] = None

        with pytest.raises(ValueError, match="inter_agent_map is None"):
            package(state, settings=_make_settings())


# ── JSON Writer Tests ────────────────────────────────────────────────


class TestJsonWriter:
    """Tests for the deterministic JSON writer."""

    def test_writes_to_correct_path(self, tmp_path):
        """write_json creates prompt_config.json in output_dir."""
        artifact = FinalOutputArtifact(
            project="Test Project",
            generated_at="2025-01-01T00:00:00Z",
            modules=[_make_artifact()],
            inter_agent_map=_make_inter_agent_map(),
            pipeline_metadata={
                "total_modules": 1, "ai_modules": 1,
                "total_estimated_tokens": 700, "average_quality_score": 8.2,
                "critic_iterations_used": 1, "total_pipeline_tokens_consumed": 500,
                "generation_duration_seconds": 5.0,
            },
        )
        result = write_json(artifact, tmp_path)

        assert result == tmp_path / "prompt_config.json"
        assert result.exists()

    def test_json_roundtrips(self, tmp_path):
        """JSON output round-trips through FinalOutputArtifact validation."""
        artifact = FinalOutputArtifact(
            project="Test Project",
            generated_at="2025-01-01T00:00:00Z",
            modules=[_make_artifact("Question Generator"), _make_artifact("Feedback Generator")],
            inter_agent_map=_make_inter_agent_map(),
            pipeline_metadata={
                "total_modules": 2, "ai_modules": 2,
                "total_estimated_tokens": 1400, "average_quality_score": 8.2,
                "critic_iterations_used": 1, "total_pipeline_tokens_consumed": 1000,
                "generation_duration_seconds": 10.0,
            },
        )
        path = write_json(artifact, tmp_path)

        raw = path.read_text(encoding="utf-8")
        restored = FinalOutputArtifact.model_validate_json(raw)
        assert restored.project == "Test Project"
        assert len(restored.modules) == 2

    def test_json_contains_module_names(self, tmp_path):
        """JSON output contains all module names."""
        artifact = FinalOutputArtifact(
            project="Test",
            generated_at="2025-01-01T00:00:00Z",
            modules=[_make_artifact("Question Generator"), _make_artifact("Feedback Generator")],
            inter_agent_map=_make_inter_agent_map(),
            pipeline_metadata={
                "total_modules": 2, "ai_modules": 2,
                "total_estimated_tokens": 1400, "average_quality_score": 6.5,
                "critic_iterations_used": 2, "total_pipeline_tokens_consumed": 2000,
                "generation_duration_seconds": 15.0,
            },
        )
        path = write_json(artifact, tmp_path)
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        module_names = [m["module_name"] for m in data["modules"]]
        assert "Question Generator" in module_names
        assert "Feedback Generator" in module_names


# ── Markdown Writer Tests ────────────────────────────────────────────


class TestMarkdownWriter:
    """Tests for the hybrid Markdown writer."""

    def _make_state_for_markdown(self):
        return _make_full_state()

    def test_writes_to_correct_path(self, tmp_path):
        """write_markdown creates architecture_spec.md."""
        state = self._make_state_for_markdown()
        path = write_markdown(state, "Executive summary.", tmp_path)

        assert path == tmp_path / "architecture_spec.md"
        assert path.exists()

    def test_contains_module_names(self, tmp_path):
        """Markdown contains all module names."""
        state = self._make_state_for_markdown()
        path = write_markdown(state, "Summary text.", tmp_path)
        content = path.read_text(encoding="utf-8")

        assert "Question Generator" in content
        assert "Feedback Generator" in content

    def test_contains_technique_info(self, tmp_path):
        """Markdown contains technique information."""
        state = self._make_state_for_markdown()
        path = write_markdown(state, "Summary.", tmp_path)
        content = path.read_text(encoding="utf-8")

        assert "chain_of_thought" in content

    def test_contains_token_budget(self, tmp_path):
        """Markdown contains token budget table."""
        state = self._make_state_for_markdown()
        path = write_markdown(state, "Summary.", tmp_path)
        content = path.read_text(encoding="utf-8")

        assert "Token Budget" in content
        assert "700" in content  # total tokens per module

    def test_contains_narrative(self, tmp_path):
        """Markdown contains the LLM-generated narrative."""
        state = self._make_state_for_markdown()
        narrative = "This is the executive summary from the LLM."
        path = write_markdown(state, narrative, tmp_path)
        content = path.read_text(encoding="utf-8")

        assert "This is the executive summary from the LLM." in content

    def test_contains_quality_scores(self, tmp_path):
        """Markdown contains quality assessment section."""
        state = self._make_state_for_markdown()
        path = write_markdown(state, "Summary.", tmp_path)
        content = path.read_text(encoding="utf-8")

        assert "Quality Assessment" in content
        assert "8.2" in content


# ── Scaffold Writer Tests ────────────────────────────────────────────


class TestScaffoldWriter:
    """Tests for the deterministic scaffold writer."""

    def _make_state_for_scaffold(self):
        return _make_full_state()

    def test_creates_directory_structure(self, tmp_path):
        """write_scaffolding creates the expected directory tree."""
        state = self._make_state_for_scaffold()
        scaffold_dir = write_scaffolding(state, tmp_path)

        assert scaffold_dir == tmp_path / "scaffolding"
        assert (scaffold_dir / "prompts").is_dir()
        assert (scaffold_dir / "agents").is_dir()
        assert (scaffold_dir / "README.md").exists()
        assert (scaffold_dir / "config.py").exists()
        assert (scaffold_dir / "main.py").exists()

    def test_prompt_files_contain_system_prompt(self, tmp_path):
        """Prompt .txt files contain the artifact's system_prompt text."""
        state = self._make_state_for_scaffold()
        scaffold_dir = write_scaffolding(state, tmp_path)

        prompt_file = scaffold_dir / "prompts" / "question_generator.txt"
        assert prompt_file.exists()
        content = prompt_file.read_text(encoding="utf-8")
        assert "Question Generator Specialist" in content

    def test_agent_stubs_are_valid_python(self, tmp_path):
        """Agent stub .py files compile without syntax errors."""
        state = self._make_state_for_scaffold()
        scaffold_dir = write_scaffolding(state, tmp_path)

        agent_files = list((scaffold_dir / "agents").glob("*_agent.py"))
        assert len(agent_files) == 2

        for agent_file in agent_files:
            # py_compile raises py_compile.PyCompileError on syntax errors
            py_compile.compile(str(agent_file), doraise=True)

    def test_readme_exists_with_project_name(self, tmp_path):
        """README.md exists and contains the project name."""
        state = self._make_state_for_scaffold()
        scaffold_dir = write_scaffolding(state, tmp_path)

        readme = scaffold_dir / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "Medical Quiz Platform" in content

    def test_snake_case_filenames(self, tmp_path):
        """Module names are correctly converted to snake_case filenames."""
        state = self._make_state_for_scaffold()
        scaffold_dir = write_scaffolding(state, tmp_path)

        assert (scaffold_dir / "prompts" / "question_generator.txt").exists()
        assert (scaffold_dir / "prompts" / "feedback_generator.txt").exists()
        assert (scaffold_dir / "agents" / "question_generator_agent.py").exists()
        assert (scaffold_dir / "agents" / "feedback_generator_agent.py").exists()

    def test_raises_without_artifacts(self, tmp_path):
        """write_scaffolding raises ValueError when no artifacts exist."""
        settings = _make_settings()
        state = create_initial_state(
            project_idea="test", config=settings.safe_dict(), run_id="test-001",
        )
        with pytest.raises(ValueError, match="no prompt_artifacts"):
            write_scaffolding(state, tmp_path)
