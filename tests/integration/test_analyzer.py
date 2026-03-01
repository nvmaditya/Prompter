"""Integration tests for the Analyzer agent with mocked LLM."""

import json
from unittest.mock import MagicMock, patch

import pytest

from prompter.agents.analyzer import analyze
from prompter.config import Settings
from prompter.models.module_map import InteractionType, ModuleMap
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

    @patch("prompter.agents.analyzer.call_llm")
    def test_generate_command_runs_analyzer(self, mock_call_llm):
        """The generate CLI command invokes the Analyzer and displays output."""
        from typer.testing import CliRunner
        from prompter.cli import app

        mock_call_llm.return_value = ModuleMap.model_validate(GOOD_MODULE_MAP_RESPONSE)

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "a quizzing platform for medical students"])

        assert result.exit_code == 0
        assert "Medical Quiz Platform" in result.output
        assert "Question Generator" in result.output

    def test_generate_rejects_short_input(self):
        """CLI rejects ideas shorter than 10 characters."""
        from typer.testing import CliRunner
        from prompter.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "hi"])

        assert result.exit_code == 1

    @patch("prompter.agents.analyzer.call_llm")
    def test_generate_clarification_exit_code(self, mock_call_llm):
        """CLI exits with code 2 when clarification is needed."""
        from typer.testing import CliRunner
        from prompter.cli import app

        mock_call_llm.return_value = ModuleMap.model_validate(VAGUE_INPUT_RESPONSE)

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "an app that does stuff"])

        assert result.exit_code == 2
