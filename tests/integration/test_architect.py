"""Integration tests for the Architect agent with mocked LLM."""

from unittest.mock import call, patch

import pytest

from prompter.agents.architect import architect
from prompter.config import Settings
from prompter.llm.techniques import TECHNIQUE_REGISTRY
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
from prompter.state import create_initial_state


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance for testing."""
    defaults = {
        "groq_api_key": "test-key",
        "groq_model": "llama-3.3-70b-versatile",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_module_map() -> ModuleMap:
    """Create a ModuleMap with 3 AI modules and 2 non-AI modules."""
    return ModuleMap(
        project_name="Medical Quiz Platform",
        domain_classification=DomainClassification(
            primary="Education", secondary=["Healthcare"]
        ),
        interaction_model=InteractionType.adaptive,
        interaction_model_rationale="Medical quizzing requires adaptive difficulty.",
        needs_clarification=False,
        clarification_questions=[],
        modules=[
            Module(
                name="Question Generator",
                description="Generates quiz questions based on medical topics and difficulty level.",
                requires_ai=True,
                ai_justification="Requires natural language generation.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.batch,
                data_inputs=["topic", "difficulty_level", "question_format"],
                data_outputs=["question", "options", "correct_answer", "explanation"],
                failure_modes=["Off-topic questions", "Medically inaccurate content"],
                depends_on=[],
            ),
            Module(
                name="User Progress Tracker",
                description="Tracks learner performance history across quiz sessions.",
                requires_ai=False,
                interaction_type=InteractionType.batch,
                data_inputs=["user_id", "quiz_results"],
                data_outputs=["performance_history", "topic_mastery_scores"],
                failure_modes=["Data loss on concurrent writes"],
                depends_on=[],
            ),
            Module(
                name="Adaptive Difficulty Engine",
                description="Adjusts question difficulty based on learner performance patterns.",
                requires_ai=True,
                ai_justification="Difficulty adaptation requires reasoning about performance patterns.",
                ai_capability_needed="analysis",
                interaction_type=InteractionType.adaptive,
                data_inputs=["performance_history", "current_session_metrics"],
                data_outputs=["recommended_difficulty", "adaptation_rationale"],
                failure_modes=["Difficulty stuck at one level", "Oscillating difficulty"],
                depends_on=["User Progress Tracker"],
            ),
            Module(
                name="Feedback Generator",
                description="Produces personalized explanations for correct and incorrect answers.",
                requires_ai=True,
                ai_justification="Requires NLG for personalized educational feedback.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.conversational,
                data_inputs=["question", "user_answer", "correct_answer", "user_level"],
                data_outputs=["feedback_text", "learning_tip"],
                failure_modes=["Generic feedback ignoring user level"],
                depends_on=["Question Generator"],
            ),
            Module(
                name="Quiz Session Manager",
                description="Orchestrates quiz flow, timing, and session state.",
                requires_ai=False,
                interaction_type=InteractionType.real_time,
                data_inputs=["user_id", "quiz_config"],
                data_outputs=["session_state", "quiz_results"],
                failure_modes=["Session timeout data loss"],
                depends_on=["Question Generator", "Adaptive Difficulty Engine"],
            ),
        ],
        module_count=5,
        ai_module_count=3,
    )


def _make_artifact(module_name: str, technique: str = "chain_of_thought") -> PromptArtifact:
    """Create a valid PromptArtifact for a given module."""
    return PromptArtifact(
        module_name=module_name,
        agent_role=f"{module_name} Specialist",
        primary_technique=technique,
        secondary_technique="role_prompting",
        technique_rationale=f"Chain-of-Thought enables step-by-step reasoning for {module_name}. Role Prompting establishes the expert persona.",
        system_prompt=(
            f"You are a {module_name} Specialist.\n\n"
            f"## Your Task\nProcess inputs and produce structured outputs for the {module_name}.\n\n"
            f"## Processing Steps\n1. Analyze the input data\n2. Apply domain knowledge\n3. Generate output\n\n"
            f"## Constraints\n- NEVER produce inaccurate information\n- ALWAYS follow the output format\n- Keep responses concise\n\n"
            f"## Output Format\n```json\n{{\"result\": \"...\", \"confidence\": 0.0}}\n```\n\n"
            f"## Context\n- Input: {{{{input_data}}}}\n- Level: {{{{user_level}}}}"
        ),
        context_slots=[
            ContextSlot(
                variable="input_data",
                description=f"Primary input data for {module_name}",
                source="Previous module output",
                injection_time="runtime",
                fallback="N/A — required",
                required=True,
            ),
            ContextSlot(
                variable="user_level",
                description="Learner proficiency level",
                source="User Progress Tracker",
                injection_time="session_start",
                fallback="intermediate",
                required=False,
            ),
        ],
        token_estimate=TokenEstimate(
            system_tokens=390,
            expected_context_tokens=300,
            expected_output_tokens=250,
            total=940,
        ),
        triggers=["module_invoked"],
        outputs_to=["Quiz Session Manager"],
        eval_criteria=EvalCriteria(
            good_output_examples=[
                f"Well-structured output from {module_name} with clear reasoning",
                f"Accurate, level-appropriate response from {module_name}",
            ],
            bad_output_examples=[
                "Empty or malformed JSON response",
                "Generic output that ignores context",
            ],
            automated_eval_suggestions=[
                "Verify JSON schema compliance",
                "Check output references specific input context",
                "Validate token count within budget",
            ],
            human_review_criteria=["Domain accuracy of generated content"],
        ),
    )


def _make_state_with_module_map() -> dict:
    """Create pipeline state with an Analyzer-produced ModuleMap."""
    settings = _make_settings()
    state = create_initial_state(
        project_idea="a quizzing platform for medical students",
        config=settings.safe_dict(),
        run_id="test-run-001",
    )
    state["module_map"] = _make_module_map()
    state["last_checkpoint"] = "analyze"
    return state


class TestArchitectWithMockedLLM:
    """Tests using mocked LLM responses (no API calls)."""

    @patch("prompter.agents.architect.call_llm")
    def test_produces_artifact_per_ai_module(self, mock_call_llm):
        """Architect produces one PromptArtifact for each AI module."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        assert len(result["prompt_artifacts"]) == 3
        assert mock_call_llm.call_count == 3

    @patch("prompter.agents.architect.call_llm")
    def test_skips_non_ai_modules(self, mock_call_llm):
        """Architect does not generate artifacts for non-AI modules."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        artifact_names = [a.module_name for a in result["prompt_artifacts"]]
        assert "User Progress Tracker" not in artifact_names
        assert "Quiz Session Manager" not in artifact_names

    @patch("prompter.agents.architect.call_llm")
    def test_artifacts_have_required_fields(self, mock_call_llm):
        """Each PromptArtifact has all required fields populated."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator", "few_shot"),
            _make_artifact("Adaptive Difficulty Engine", "chain_of_thought"),
            _make_artifact("Feedback Generator", "role_prompting"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            assert artifact.module_name
            assert artifact.agent_role
            assert artifact.primary_technique
            assert artifact.technique_rationale
            assert artifact.system_prompt
            assert len(artifact.context_slots) > 0
            assert artifact.token_estimate.total > 0
            assert len(artifact.triggers) > 0
            assert len(artifact.outputs_to) > 0

    @patch("prompter.agents.architect.call_llm")
    def test_techniques_from_valid_set(self, mock_call_llm):
        """All technique selections come from the technique registry."""
        valid_keys = set(TECHNIQUE_REGISTRY.keys())
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator", "few_shot"),
            _make_artifact("Adaptive Difficulty Engine", "chain_of_thought"),
            _make_artifact("Feedback Generator", "role_prompting"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            assert artifact.primary_technique in valid_keys, (
                f"Unknown technique: {artifact.primary_technique}"
            )
            if artifact.secondary_technique:
                assert artifact.secondary_technique in valid_keys

    @patch("prompter.agents.architect.call_llm")
    def test_context_slots_use_template_syntax(self, mock_call_llm):
        """Context slots use {{variable}} syntax in the system prompt."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            for slot in artifact.context_slots:
                placeholder = "{{" + slot.variable + "}}"
                assert placeholder in artifact.system_prompt, (
                    f"Slot {slot.variable} not found as {placeholder} in system prompt"
                )

    @patch("prompter.agents.architect.call_llm")
    def test_eval_criteria_meets_minimums(self, mock_call_llm):
        """EvalCriteria has >=2 good/bad examples, >=3 eval suggestions, >=1 human criteria."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            ec = artifact.eval_criteria
            assert len(ec.good_output_examples) >= 2
            assert len(ec.bad_output_examples) >= 2
            assert len(ec.automated_eval_suggestions) >= 3
            assert len(ec.human_review_criteria) >= 1

    @patch("prompter.agents.architect.call_llm")
    def test_telemetry_populated(self, mock_call_llm):
        """Architect populates token_usage and agent_durations."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        assert "architect" in result["agent_durations"]
        assert result["agent_durations"]["architect"] >= 0
        assert result["last_checkpoint"] == "architect"

    @patch("prompter.agents.architect.call_llm")
    def test_pydantic_roundtrip(self, mock_call_llm):
        """Each PromptArtifact round-trips through Pydantic serialization."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        result = architect(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            json_str = artifact.model_dump_json()
            restored = PromptArtifact.model_validate_json(json_str)
            assert restored.module_name == artifact.module_name
            assert restored.primary_technique == artifact.primary_technique
            assert len(restored.eval_criteria.good_output_examples) == len(
                artifact.eval_criteria.good_output_examples
            )

    @patch("prompter.agents.architect.call_llm")
    def test_system_prompt_includes_technique_catalog(self, mock_call_llm):
        """Architect injects the technique catalog into the system prompt."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        architect(state, settings=_make_settings())

        # Verify the system prompt passed to call_llm contains technique catalog content
        first_call_args = mock_call_llm.call_args_list[0]
        system_prompt_used = first_call_args.kwargs.get(
            "system_prompt", first_call_args.args[0] if first_call_args.args else ""
        )
        assert "Chain-of-Thought" in system_prompt_used
        assert "Few-Shot" in system_prompt_used
        assert "Role Prompting" in system_prompt_used
        assert "{{TECHNIQUE_CATALOG}}" not in system_prompt_used

    @patch("prompter.agents.architect.call_llm")
    def test_user_message_contains_module_map_and_target(self, mock_call_llm):
        """Each call_llm receives the full ModuleMap and the specific target module."""
        mock_call_llm.side_effect = [
            _make_artifact("Question Generator"),
            _make_artifact("Adaptive Difficulty Engine"),
            _make_artifact("Feedback Generator"),
        ]

        state = _make_state_with_module_map()
        architect(state, settings=_make_settings())

        # Check first call targets Question Generator
        first_call_args = mock_call_llm.call_args_list[0]
        user_msg = first_call_args.kwargs.get(
            "user_message", first_call_args.args[1] if len(first_call_args.args) > 1 else ""
        )
        assert "Medical Quiz Platform" in user_msg  # Full ModuleMap context
        assert "Question Generator" in user_msg  # Target module

    def test_raises_without_module_map(self):
        """Architect raises ValueError when module_map is None."""
        settings = _make_settings()
        state = create_initial_state(
            project_idea="test",
            config=settings.safe_dict(),
            run_id="test-run-001",
        )

        with pytest.raises(ValueError, match="module_map is None"):
            architect(state, settings=settings)
