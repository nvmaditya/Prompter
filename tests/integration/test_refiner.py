"""Integration tests for the Refiner agent with mocked LLM."""

from unittest.mock import patch

import pytest

from prompter.agents.refiner import refine
from prompter.config import Settings
from prompter.models.critic_feedback import (
    CategoryScore,
    CriticFeedback,
    Issue,
    IssueCategory,
    Severity,
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
        technique_rationale=f"CoT for {module_name}.",
        system_prompt=f"You are a {module_name} Specialist.\n\nInput: {{{{input_data}}}}\nLevel: {{{{user_level}}}}",
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


def _make_revised_artifact(module_name: str) -> PromptArtifact:
    """Create a revised artifact (slightly different system_prompt)."""
    return PromptArtifact(
        module_name=module_name,
        agent_role=f"{module_name} Specialist",
        primary_technique="chain_of_thought",
        technique_rationale=f"CoT for {module_name}.",
        system_prompt=(
            f"You are a {module_name} Specialist.\n\n"
            f"## Constraints\n- NEVER fabricate information\n- ALWAYS cite sources\n\n"
            f"Input: {{{{input_data}}}}\nLevel: {{{{user_level}}}}"
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
            system_tokens=350, expected_context_tokens=200,
            expected_output_tokens=200, total=750,
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


def _make_passing_feedback(module_name: str) -> CriticFeedback:
    return CriticFeedback(
        module_name=module_name,
        overall_score=8.2,
        passed=True,
        category_scores={
            "ambiguity": CategoryScore(score=8.5),
            "hallucination_risk": CategoryScore(score=8.0),
            "missing_constraints": CategoryScore(score=8.0),
            "edge_cases": CategoryScore(score=8.0),
            "token_efficiency": CategoryScore(score=8.5),
        },
        issues=[],
        iteration=1,
        summary="Strong prompt.",
    )


def _make_failing_feedback(module_name: str) -> CriticFeedback:
    return CriticFeedback(
        module_name=module_name,
        overall_score=5.5,
        passed=False,
        category_scores={
            "ambiguity": CategoryScore(score=6.0),
            "hallucination_risk": CategoryScore(score=4.0, issues=[
                Issue(
                    category=IssueCategory.hallucination_risk,
                    severity=Severity.high,
                    location="Include an explanation",
                    description="No grounding constraints.",
                    suggestion="Add grounding rules.",
                ),
            ]),
            "missing_constraints": CategoryScore(score=6.0),
            "edge_cases": CategoryScore(score=5.0),
            "token_efficiency": CategoryScore(score=7.0),
        },
        issues=[
            Issue(
                category=IssueCategory.hallucination_risk,
                severity=Severity.high,
                location="Include an explanation",
                description="No grounding constraints.",
                suggestion="Add grounding rules.",
            ),
        ],
        iteration=1,
        summary="Lacks hallucination prevention.",
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
            Module(
                name="Adaptive Difficulty Engine",
                description="Adjusts difficulty.",
                requires_ai=True,
                ai_justification="Pattern analysis.",
                ai_capability_needed="analysis",
                interaction_type=InteractionType.adaptive,
                data_inputs=["performance_history"],
                data_outputs=["recommended_difficulty"],
                failure_modes=["Stuck"],
            ),
        ],
        module_count=3,
        ai_module_count=3,
    )


def _make_state_with_feedback(all_pass: bool = False) -> dict:
    """Create state with artifacts and one round of critic feedback."""
    settings = _make_settings()
    state = create_initial_state(
        project_idea="a quizzing platform",
        config=settings.safe_dict(),
        run_id="test-run-001",
    )
    state["module_map"] = _make_module_map()
    state["prompt_artifacts"] = [
        _make_artifact("Question Generator"),
        _make_artifact("Feedback Generator"),
        _make_artifact("Adaptive Difficulty Engine"),
    ]

    if all_pass:
        state["critic_feedback"] = [[
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]]
    else:
        # Question Generator fails, others pass
        state["critic_feedback"] = [[
            _make_failing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]]
    state["last_checkpoint"] = "critique"
    return state


class TestRefinerWithMockedLLM:
    """Tests using mocked LLM responses."""

    @patch("prompter.agents.refiner.call_llm")
    def test_only_revises_failed_artifacts(self, mock_call_llm):
        """Refiner only calls LLM for failing artifacts, keeps passing ones."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        result = refine(state, settings=_make_settings())

        # Only 1 LLM call (Question Generator failed, 2 others passed)
        assert mock_call_llm.call_count == 1
        assert len(result["prompt_artifacts"]) == 3

    @patch("prompter.agents.refiner.call_llm")
    def test_revised_artifact_replaces_original(self, mock_call_llm):
        """The revised artifact appears in the output list at the correct position."""
        revised = _make_revised_artifact("Question Generator")
        mock_call_llm.return_value = revised

        state = _make_state_with_feedback(all_pass=False)
        result = refine(state, settings=_make_settings())

        # First artifact should be the revised one
        first = result["prompt_artifacts"][0]
        assert "NEVER fabricate" in first.system_prompt

    @patch("prompter.agents.refiner.call_llm")
    def test_passing_artifacts_preserved(self, mock_call_llm):
        """Passing artifacts are kept unchanged."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        original_feedback = state["prompt_artifacts"][1]  # Feedback Generator (passing)
        result = refine(state, settings=_make_settings())

        assert result["prompt_artifacts"][1].module_name == "Feedback Generator"
        assert result["prompt_artifacts"][1].system_prompt == original_feedback.system_prompt

    @patch("prompter.agents.refiner.call_llm")
    def test_current_iteration_incremented(self, mock_call_llm):
        """current_iteration is incremented by 1."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        state["current_iteration"] = 0
        result = refine(state, settings=_make_settings())

        assert result["current_iteration"] == 1

    @patch("prompter.agents.refiner.call_llm")
    def test_no_llm_calls_when_all_pass(self, mock_call_llm):
        """When all artifacts passed, refiner makes no LLM calls."""
        state = _make_state_with_feedback(all_pass=True)
        result = refine(state, settings=_make_settings())

        mock_call_llm.assert_not_called()
        assert len(result["prompt_artifacts"]) == 3
        assert result["current_iteration"] == 1

    @patch("prompter.agents.refiner.call_llm")
    def test_user_message_contains_artifact_and_feedback(self, mock_call_llm):
        """User message includes original artifact and critic feedback."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        refine(state, settings=_make_settings())

        call_args = mock_call_llm.call_args_list[0]
        user_msg = call_args.kwargs.get(
            "user_message", call_args.args[1] if len(call_args.args) > 1 else ""
        )
        assert "Question Generator" in user_msg
        assert "Critic Feedback" in user_msg
        assert "hallucination_risk" in user_msg

    @patch("prompter.agents.refiner.call_llm")
    def test_telemetry_populated(self, mock_call_llm):
        """Refiner populates agent_durations and last_checkpoint."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        result = refine(state, settings=_make_settings())

        assert "refiner" in result["agent_durations"]
        assert result["agent_durations"]["refiner"] >= 0
        assert result["last_checkpoint"] == "refine"

    @patch("prompter.agents.refiner.call_llm")
    def test_revised_artifact_pydantic_roundtrip(self, mock_call_llm):
        """Revised PromptArtifact round-trips through serialization."""
        mock_call_llm.return_value = _make_revised_artifact("Question Generator")

        state = _make_state_with_feedback(all_pass=False)
        result = refine(state, settings=_make_settings())

        for artifact in result["prompt_artifacts"]:
            json_str = artifact.model_dump_json()
            restored = PromptArtifact.model_validate_json(json_str)
            assert restored.module_name == artifact.module_name

    def test_raises_without_critic_feedback(self):
        """Refiner raises ValueError when critic_feedback is empty."""
        settings = _make_settings()
        state = create_initial_state(
            project_idea="test",
            config=settings.safe_dict(),
            run_id="test-run-001",
        )
        state["prompt_artifacts"] = [_make_artifact()]
        with pytest.raises(ValueError, match="critic_feedback is empty"):
            refine(state, settings=settings)

    @patch("prompter.agents.refiner.call_llm")
    def test_multiple_failures_all_revised(self, mock_call_llm):
        """When multiple artifacts fail, all are revised."""
        mock_call_llm.side_effect = [
            _make_revised_artifact("Question Generator"),
            _make_revised_artifact("Feedback Generator"),
        ]

        state = _make_state_with_feedback(all_pass=False)
        # Make Feedback Generator also fail
        state["critic_feedback"][0][1] = _make_failing_feedback("Feedback Generator")

        result = refine(state, settings=_make_settings())

        assert mock_call_llm.call_count == 2
        assert len(result["prompt_artifacts"]) == 3
