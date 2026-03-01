"""Integration tests for the Critic agent with mocked LLM."""

from unittest.mock import patch

import pytest

from prompter.agents.critic import critique
from prompter.config import Settings
from prompter.models.critic_feedback import (
    CategoryScore,
    CriticFeedback,
    Issue,
    IssueCategory,
    Severity,
)
from prompter.models.inter_agent_map import InterAgentMap
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
        technique_rationale=f"CoT for step-by-step reasoning in {module_name}.",
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
            system_tokens=300,
            expected_context_tokens=200,
            expected_output_tokens=200,
            total=700,
        ),
        triggers=["module_invoked"],
        outputs_to=["Quiz Session Manager"],
        eval_criteria=EvalCriteria(
            good_output_examples=["Good output 1", "Good output 2"],
            bad_output_examples=["Bad output 1", "Bad output 2"],
            automated_eval_suggestions=["Check JSON", "Verify references", "Validate tokens"],
            human_review_criteria=["Domain accuracy"],
        ),
    )


def _make_passing_feedback(module_name: str, iteration: int = 1) -> CriticFeedback:
    """Create a CriticFeedback that passes (score >= 7.0)."""
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
        iteration=iteration,
        summary="Strong prompt with clear instructions and good constraints.",
    )


def _make_failing_feedback(module_name: str, iteration: int = 1) -> CriticFeedback:
    """Create a CriticFeedback that fails (score < 7.0)."""
    return CriticFeedback(
        module_name=module_name,
        overall_score=5.5,
        passed=False,
        category_scores={
            "ambiguity": CategoryScore(
                score=6.0,
                issues=[
                    Issue(
                        category=IssueCategory.ambiguity,
                        severity=Severity.medium,
                        location="Generate questions based on the topic",
                        description="Vague topic relevance instruction.",
                        suggestion="Specify 'directly test knowledge of the topic'.",
                    )
                ],
            ),
            "hallucination_risk": CategoryScore(
                score=4.0,
                issues=[
                    Issue(
                        category=IssueCategory.hallucination_risk,
                        severity=Severity.high,
                        location="Include an explanation for the correct answer",
                        description="No grounding for medical explanations.",
                        suggestion="Add: 'All explanations must be consistent with medical consensus.'",
                    )
                ],
            ),
            "missing_constraints": CategoryScore(score=6.0),
            "edge_cases": CategoryScore(score=5.0),
            "token_efficiency": CategoryScore(score=7.0),
        },
        issues=[
            Issue(
                category=IssueCategory.ambiguity,
                severity=Severity.medium,
                location="Generate questions based on the topic",
                description="Vague topic relevance instruction.",
                suggestion="Specify 'directly test knowledge of the topic'.",
            ),
            Issue(
                category=IssueCategory.hallucination_risk,
                severity=Severity.high,
                location="Include an explanation for the correct answer",
                description="No grounding for medical explanations.",
                suggestion="Add: 'All explanations must be consistent with medical consensus.'",
            ),
        ],
        iteration=iteration,
        summary="Prompt lacks hallucination prevention and has ambiguous instructions.",
    )


def _make_module_map() -> ModuleMap:
    return ModuleMap(
        project_name="Medical Quiz Platform",
        domain_classification=DomainClassification(primary="Education", secondary=["Healthcare"]),
        interaction_model=InteractionType.adaptive,
        interaction_model_rationale="Adaptive quiz difficulty.",
        modules=[
            Module(
                name="Question Generator",
                description="Generates quiz questions.",
                requires_ai=True,
                ai_justification="Requires NLG.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.batch,
                data_inputs=["topic", "difficulty_level"],
                data_outputs=["question", "options"],
                failure_modes=["Off-topic"],
            ),
            Module(
                name="Feedback Generator",
                description="Produces answer feedback.",
                requires_ai=True,
                ai_justification="Requires NLG.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.conversational,
                data_inputs=["question", "user_answer"],
                data_outputs=["feedback_text"],
                failure_modes=["Generic feedback"],
                depends_on=["Question Generator"],
            ),
            Module(
                name="Adaptive Difficulty Engine",
                description="Adjusts difficulty.",
                requires_ai=True,
                ai_justification="Pattern reasoning.",
                ai_capability_needed="analysis",
                interaction_type=InteractionType.adaptive,
                data_inputs=["performance_history"],
                data_outputs=["recommended_difficulty"],
                failure_modes=["Stuck difficulty"],
            ),
        ],
        module_count=3,
        ai_module_count=3,
    )


def _make_state_with_artifacts() -> dict:
    settings = _make_settings()
    state = create_initial_state(
        project_idea="a quizzing platform for medical students",
        config=settings.safe_dict(),
        run_id="test-run-001",
    )
    state["module_map"] = _make_module_map()
    state["prompt_artifacts"] = [
        _make_artifact("Question Generator"),
        _make_artifact("Feedback Generator"),
        _make_artifact("Adaptive Difficulty Engine"),
    ]
    state["last_checkpoint"] = "design_communication"
    return state


EXPECTED_CATEGORY_KEYS = {
    "ambiguity",
    "hallucination_risk",
    "missing_constraints",
    "edge_cases",
    "token_efficiency",
}


class TestCriticWithMockedLLM:
    """Tests using mocked LLM responses."""

    @patch("prompter.agents.critic.call_llm")
    def test_all_pass_when_scores_above_threshold(self, mock_call_llm):
        """All artifacts passing produces all_passed=True."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        assert result["all_passed"] is True
        assert len(result["critic_feedback"]) == 1  # One iteration
        assert len(result["critic_feedback"][0]) == 3  # 3 modules

    @patch("prompter.agents.critic.call_llm")
    def test_fails_when_any_score_below_threshold(self, mock_call_llm):
        """One failing artifact produces all_passed=False."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_failing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        assert result["all_passed"] is False

    @patch("prompter.agents.critic.call_llm")
    def test_category_scores_have_all_five_keys(self, mock_call_llm):
        """Each CriticFeedback has all 5 scoring dimensions."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_failing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        for feedback in result["critic_feedback"][0]:
            assert set(feedback.category_scores.keys()) == EXPECTED_CATEGORY_KEYS

    @patch("prompter.agents.critic.call_llm")
    def test_failing_feedback_has_issues(self, mock_call_llm):
        """Failing feedback includes non-empty issues with required fields."""
        mock_call_llm.side_effect = [
            _make_failing_feedback("Question Generator"),
            _make_failing_feedback("Feedback Generator"),
            _make_failing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        for feedback in result["critic_feedback"][0]:
            assert len(feedback.issues) > 0
            for issue in feedback.issues:
                assert issue.category in IssueCategory
                assert issue.severity in Severity
                assert issue.location  # Quoted text
                assert issue.description
                assert issue.suggestion

    @patch("prompter.agents.critic.call_llm")
    def test_best_prompt_versions_updated(self, mock_call_llm):
        """best_prompt_versions is populated for each module."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_failing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        bpv = result["best_prompt_versions"]
        assert "Question Generator" in bpv
        assert "Feedback Generator" in bpv
        assert "Adaptive Difficulty Engine" in bpv
        assert bpv["Question Generator"]["score"] == 8.2
        assert bpv["Feedback Generator"]["score"] == 5.5

    @patch("prompter.agents.critic.call_llm")
    def test_best_versions_keep_higher_score(self, mock_call_llm):
        """On second iteration, best_prompt_versions keeps the higher score."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        # Simulate a prior iteration with a lower score
        state["best_prompt_versions"] = {
            "Question Generator": {
                "artifact": _make_artifact("Question Generator").model_dump(),
                "score": 6.0,
            },
        }
        result = critique(state, settings=_make_settings())

        # New score (8.2) should replace old score (6.0)
        assert result["best_prompt_versions"]["Question Generator"]["score"] == 8.2

    @patch("prompter.agents.critic.call_llm")
    def test_feedback_appended_as_iteration(self, mock_call_llm):
        """critic_feedback is a nested list — new iteration appended."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        # Simulate prior iteration
        state["critic_feedback"] = [
            [
                _make_failing_feedback("Question Generator"),
                _make_failing_feedback("Feedback Generator"),
                _make_failing_feedback("Adaptive Difficulty Engine"),
            ]
        ]

        result = critique(state, settings=_make_settings())

        assert len(result["critic_feedback"]) == 2  # Two iterations now
        assert len(result["critic_feedback"][0]) == 3  # First iteration
        assert len(result["critic_feedback"][1]) == 3  # Second iteration

    @patch("prompter.agents.critic.call_llm")
    def test_telemetry_populated(self, mock_call_llm):
        """Critic populates agent_durations and last_checkpoint."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        assert "critic" in result["agent_durations"]
        assert result["agent_durations"]["critic"] >= 0
        assert result["last_checkpoint"] == "critique"

    @patch("prompter.agents.critic.call_llm")
    def test_pydantic_roundtrip(self, mock_call_llm):
        """CriticFeedback round-trips through serialization."""
        mock_call_llm.side_effect = [
            _make_failing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        result = critique(state, settings=_make_settings())

        for feedback in result["critic_feedback"][0]:
            json_str = feedback.model_dump_json()
            restored = CriticFeedback.model_validate_json(json_str)
            assert restored.module_name == feedback.module_name
            assert restored.overall_score == feedback.overall_score

    @patch("prompter.agents.critic.call_llm")
    def test_user_message_contains_artifact_and_context(self, mock_call_llm):
        """User message includes the artifact JSON and module map."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        critique(state, settings=_make_settings())

        first_call = mock_call_llm.call_args_list[0]
        user_msg = first_call.kwargs.get(
            "user_message", first_call.args[1] if len(first_call.args) > 1 else ""
        )
        assert "Question Generator" in user_msg
        assert "PromptArtifact" in user_msg
        assert "ModuleMap" in user_msg

    @patch("prompter.agents.critic.call_llm")
    def test_system_prompt_has_threshold_injected(self, mock_call_llm):
        """System prompt has quality_threshold value injected."""
        mock_call_llm.side_effect = [
            _make_passing_feedback("Question Generator"),
            _make_passing_feedback("Feedback Generator"),
            _make_passing_feedback("Adaptive Difficulty Engine"),
        ]

        state = _make_state_with_artifacts()
        critique(state, settings=_make_settings())

        first_call = mock_call_llm.call_args_list[0]
        sys_prompt = first_call.kwargs.get(
            "system_prompt", first_call.args[0] if first_call.args else ""
        )
        assert "7.0" in sys_prompt
        assert "{{quality_threshold}}" not in sys_prompt

    def test_raises_without_prompt_artifacts(self):
        """Critic raises ValueError when prompt_artifacts is empty."""
        settings = _make_settings()
        state = create_initial_state(
            project_idea="test",
            config=settings.safe_dict(),
            run_id="test-run-001",
        )
        with pytest.raises(ValueError, match="prompt_artifacts is empty"):
            critique(state, settings=settings)
