"""Integration tests for the Analyzer agent with mocked LLM."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompter.agents.analyzer import analyze
from prompter.agents.packager import NarrativeResponse
from prompter.config import Settings
from prompter.models.critic_feedback import CategoryScore, CriticFeedback
from prompter.models.inter_agent_map import (
    HandoffCondition,
    InterAgentMap,
    SharedMemoryField,
    Trigger,
)
from prompter.models.module_map import InteractionType, ModuleMap
from prompter.models.prompt_artifact import (
    ContextSlot,
    EvalCriteria,
    PromptArtifact,
    TokenEstimate,
)
from prompter.state import create_initial_state


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance for testing."""
    defaults = {
        "groq_api_key": "test-key",
        "groq_model": "llama-3.3-70b-versatile",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_state(idea: str = "a quizzing platform for medical students") -> dict:
    """Create initial pipeline state for testing."""
    settings = _make_settings()
    return create_initial_state(
        project_idea=idea,
        config=settings.safe_dict(),
        run_id="test-run-001",
    )


GOOD_MODULE_MAP_RESPONSE = {
    "project_name": "Medical Quiz Platform",
    "domain_classification": {
        "primary": "Education",
        "secondary": ["Healthcare"],
    },
    "interaction_model": "adaptive",
    "interaction_model_rationale": "Medical quizzing requires adaptive difficulty to match learner level.",
    "needs_clarification": False,
    "clarification_questions": [],
    "modules": [
        {
            "name": "Question Generator",
            "description": "Generates quiz questions based on medical topics and difficulty level.",
            "requires_ai": True,
            "ai_justification": "Requires natural language generation to create contextually relevant medical questions.",
            "ai_capability_needed": "generation",
            "interaction_type": "batch",
            "data_inputs": ["topic", "difficulty_level", "question_format"],
            "data_outputs": ["question", "options", "correct_answer", "explanation"],
            "failure_modes": ["Off-topic questions", "Medically inaccurate content"],
            "depends_on": [],
        },
        {
            "name": "User Progress Tracker",
            "description": "Tracks learner performance history across quiz sessions.",
            "requires_ai": False,
            "ai_justification": None,
            "ai_capability_needed": None,
            "interaction_type": "batch",
            "data_inputs": ["user_id", "quiz_results"],
            "data_outputs": ["performance_history", "topic_mastery_scores"],
            "failure_modes": ["Data loss on concurrent writes"],
            "depends_on": [],
        },
        {
            "name": "Adaptive Difficulty Engine",
            "description": "Adjusts question difficulty based on learner performance patterns.",
            "requires_ai": True,
            "ai_justification": "Difficulty adaptation requires reasoning about multi-variable performance patterns.",
            "ai_capability_needed": "analysis",
            "interaction_type": "adaptive",
            "data_inputs": ["performance_history", "current_session_metrics"],
            "data_outputs": ["recommended_difficulty", "adaptation_rationale"],
            "failure_modes": ["Difficulty stuck at one level", "Oscillating difficulty"],
            "depends_on": ["User Progress Tracker"],
        },
        {
            "name": "Feedback Generator",
            "description": "Produces personalized explanations for correct and incorrect answers.",
            "requires_ai": True,
            "ai_justification": "Requires natural language generation for personalized educational feedback.",
            "ai_capability_needed": "generation",
            "interaction_type": "conversational",
            "data_inputs": ["question", "user_answer", "correct_answer", "user_level"],
            "data_outputs": ["feedback_text", "learning_tip"],
            "failure_modes": ["Generic feedback ignoring user level", "Medically inaccurate explanations"],
            "depends_on": ["Question Generator"],
        },
        {
            "name": "Quiz Session Manager",
            "description": "Orchestrates quiz flow, timing, and session state.",
            "requires_ai": False,
            "ai_justification": None,
            "ai_capability_needed": None,
            "interaction_type": "real_time",
            "data_inputs": ["user_id", "quiz_config"],
            "data_outputs": ["session_state", "quiz_results"],
            "failure_modes": ["Session timeout data loss"],
            "depends_on": ["Question Generator", "Adaptive Difficulty Engine"],
        },
    ],
    "module_count": 5,
    "ai_module_count": 3,
}


VAGUE_INPUT_RESPONSE = {
    "project_name": "Unclear Project",
    "domain_classification": {
        "primary": "Unknown",
        "secondary": [],
    },
    "interaction_model": "batch",
    "interaction_model_rationale": "Cannot determine interaction model without more detail.",
    "needs_clarification": True,
    "clarification_questions": [
        "What specific problem does this app solve?",
        "Who are the target users?",
        "What are the core features you envision?",
    ],
    "modules": [],
    "module_count": 0,
    "ai_module_count": 0,
}


class TestAnalyzerWithMockedLLM:
    """Tests using mocked LLM responses (no API calls)."""

    @patch("prompter.agents.analyzer.call_llm")
    def test_good_input_produces_valid_module_map(self, mock_call_llm):
        """Analyzer produces a valid ModuleMap for a well-defined idea."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform for medical students with adaptive difficulty")
        settings = _make_settings()
        result = analyze(state, settings=settings)

        module_map = result["module_map"]
        assert isinstance(module_map, ModuleMap)
        assert module_map.project_name == "Medical Quiz Platform"
        assert module_map.module_count == 5
        assert module_map.ai_module_count == 3
        assert not result["needs_clarification"]

    @patch("prompter.agents.analyzer.call_llm")
    def test_domain_classification(self, mock_call_llm):
        """Analyzer correctly classifies domains."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform for medical students")
        result = analyze(state, settings=_make_settings())

        domain = result["module_map"].domain_classification
        assert domain.primary == "Education"
        assert "Healthcare" in domain.secondary

    @patch("prompter.agents.analyzer.call_llm")
    def test_modules_have_required_fields(self, mock_call_llm):
        """Each module has all required fields populated."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform")
        result = analyze(state, settings=_make_settings())

        for module in result["module_map"].modules:
            assert module.name
            assert module.description
            assert isinstance(module.requires_ai, bool)
            assert len(module.data_inputs) > 0
            assert len(module.data_outputs) > 0
            assert len(module.failure_modes) > 0
            if module.requires_ai:
                assert module.ai_justification is not None
                assert module.ai_capability_needed is not None

    @patch("prompter.agents.analyzer.call_llm")
    def test_vague_input_triggers_clarification(self, mock_call_llm):
        """Vague input sets needs_clarification and provides questions."""
        mock_call_llm.return_value = ModuleMap.model_validate(VAGUE_INPUT_RESPONSE)

        state = _make_state("an app")
        result = analyze(state, settings=_make_settings())

        assert result["needs_clarification"] is True
        assert len(result["clarification_questions"]) >= 2

    @patch("prompter.agents.analyzer.call_llm")
    def test_telemetry_populated(self, mock_call_llm):
        """Analyzer populates token_usage and agent_durations."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform")
        result = analyze(state, settings=_make_settings())

        assert "analyzer" in result["agent_durations"]
        assert result["agent_durations"]["analyzer"] >= 0
        assert result["last_checkpoint"] == "analyze"

    @patch("prompter.agents.analyzer.call_llm")
    def test_interaction_model_set(self, mock_call_llm):
        """Analyzer sets the overall interaction model."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform with adaptive difficulty")
        result = analyze(state, settings=_make_settings())

        assert result["module_map"].interaction_model == InteractionType.adaptive
        assert result["module_map"].interaction_model_rationale

    @patch("prompter.agents.analyzer.call_llm")
    def test_module_map_passes_pydantic_validation(self, mock_call_llm):
        """The returned ModuleMap round-trips through Pydantic validation."""
        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        state = _make_state("a quizzing platform")
        result = analyze(state, settings=_make_settings())

        # Serialize and deserialize to verify full validation
        json_str = result["module_map"].model_dump_json()
        restored = ModuleMap.model_validate_json(json_str)
        assert restored.module_count == result["module_map"].module_count


class TestAnalyzerCLIIntegration:
    """Tests for CLI-level integration of the Analyzer."""

    @staticmethod
    def _make_artifact(module_name: str) -> PromptArtifact:
        return PromptArtifact(
            module_name=module_name,
            agent_role=f"{module_name} Specialist",
            primary_technique="chain_of_thought",
            technique_rationale="CoT for reasoning.",
            system_prompt=f"You are a {module_name} specialist.",
            context_slots=[
                ContextSlot(
                    variable="input_data", description="Input",
                    source="previous", injection_time="runtime",
                    fallback="N/A", required=True,
                ),
            ],
            token_estimate=TokenEstimate(
                system_tokens=100, expected_context_tokens=100,
                expected_output_tokens=100, total=300,
            ),
            triggers=["invoked"],
            outputs_to=["next_module"],
            eval_criteria=EvalCriteria(
                good_output_examples=["Good 1", "Good 2"],
                bad_output_examples=["Bad 1", "Bad 2"],
                automated_eval_suggestions=["Check 1", "Check 2", "Check 3"],
                human_review_criteria=["Accuracy"],
            ),
        )

    @staticmethod
    def _make_inter_agent_map() -> InterAgentMap:
        return InterAgentMap(
            shared_memory_schema={
                "topic": SharedMemoryField(
                    type="string", description="Topic",
                    written_by=["Question Generator"], read_by=["Feedback Generator"],
                    updated_on="question_generated", default="",
                ),
            },
            handoff_conditions=[
                HandoffCondition(
                    from_agent="Question Generator", to_agent="Feedback Generator",
                    condition="done", data_passed={"q": "text"}, format="json",
                    fallback_if_incomplete="retry",
                ),
            ],
            context_pollution_rules=[],
            trigger_map=[
                Trigger(
                    event="start", activates=["Question Generator"],
                    priority_order=["Question Generator"],
                    execution="sequential", error_fallback="retry",
                ),
            ],
        )

    @staticmethod
    def _make_passing_feedback(module_name: str) -> CriticFeedback:
        return CriticFeedback(
            module_name=module_name,
            overall_score=8.5,
            passed=True,
            category_scores={
                "ambiguity": CategoryScore(score=8.5),
                "hallucination_risk": CategoryScore(score=8.5),
                "missing_constraints": CategoryScore(score=8.5),
                "edge_cases": CategoryScore(score=8.5),
                "token_efficiency": CategoryScore(score=8.5),
            },
            issues=[],
            iteration=1,
            summary=f"Good quality for {module_name}.",
        )

    @patch("prompter.utils.checkpoint.save_checkpoint")
    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_generate_command_runs_analyzer(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_packager, mock_json, mock_md, mock_scaffold, mock_checkpoint,
    ):
        """The generate CLI command runs the full pipeline and completes."""
        from typer.testing import CliRunner
        from prompter.cli import app

        # AI module names from the good module map
        ai_names = ["Question Generator", "Adaptive Difficulty Engine", "Feedback Generator"]

        mock_analyze.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)
        mock_architect.side_effect = [self._make_artifact(n) for n in ai_names]
        mock_comm.return_value = self._make_inter_agent_map()
        mock_critic.side_effect = [self._make_passing_feedback(n) for n in ai_names]
        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")
        mock_checkpoint.return_value = Path(".prompter_state/test/pipeline_state.json")

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "a quizzing platform for medical students"])

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"

    def test_generate_rejects_short_input(self):
        """CLI rejects ideas shorter than 10 characters."""
        from typer.testing import CliRunner
        from prompter.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "hi"])

        assert result.exit_code == 1

    @patch("prompter.utils.checkpoint.save_checkpoint")
    @patch("prompter.agents.analyzer.call_llm")
    def test_generate_clarification_exit_code(self, mock_call_llm, mock_checkpoint):
        """CLI exits with code 2 when clarification is needed."""
        from typer.testing import CliRunner
        from prompter.cli import app

        mock_call_llm.return_value = ModuleMap.model_validate(VAGUE_INPUT_RESPONSE)
        mock_checkpoint.return_value = Path(".prompter_state/test/pipeline_state.json")

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "an app that does stuff"])

        assert result.exit_code == 2
