"""Shared test fixtures — mock LLM client and sample data."""

import json
from unittest.mock import MagicMock

import pytest

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
from prompter.models.critic_feedback import (
    CategoryScore,
    CriticFeedback,
    Issue,
    IssueCategory,
    Severity,
)
from prompter.models.inter_agent_map import (
    ContextPollutionRule,
    HandoffCondition,
    InterAgentMap,
    SharedMemoryField,
    Trigger,
)


@pytest.fixture
def sample_module() -> Module:
    return Module(
        name="Question Generator",
        description="Generates quiz questions based on medical topics.",
        requires_ai=True,
        ai_justification="Requires natural language generation.",
        ai_capability_needed="generation",
        interaction_type=InteractionType.batch,
        data_inputs=["topic", "difficulty_level"],
        data_outputs=["question", "options", "correct_answer"],
        failure_modes=["Generates off-topic questions", "Wrong difficulty level"],
        depends_on=[],
    )


@pytest.fixture
def sample_module_map(sample_module: Module) -> ModuleMap:
    non_ai_module = Module(
        name="Score Tracker",
        description="Tracks user scores over time.",
        requires_ai=False,
        interaction_type=InteractionType.batch,
        data_inputs=["user_id", "score"],
        data_outputs=["score_history"],
        failure_modes=["Data loss"],
    )
    return ModuleMap(
        project_name="Medical Quiz Platform",
        domain_classification=DomainClassification(
            primary="Education", secondary=["Healthcare"]
        ),
        interaction_model=InteractionType.adaptive,
        interaction_model_rationale="Medical quizzing requires adaptive difficulty.",
        modules=[sample_module, non_ai_module],
        module_count=2,
        ai_module_count=1,
    )


@pytest.fixture
def sample_prompt_artifact() -> PromptArtifact:
    return PromptArtifact(
        module_name="Question Generator",
        agent_role="Medical Quiz Question Author",
        primary_technique="few_shot",
        technique_rationale="Ensures consistent question format via examples.",
        system_prompt="You are a medical quiz question author...",
        context_slots=[
            ContextSlot(
                variable="topic",
                description="The medical topic for the question.",
                source="user_input",
                injection_time="session_start",
                fallback="General Medicine",
                required=True,
            )
        ],
        token_estimate=TokenEstimate(
            system_tokens=500,
            expected_context_tokens=200,
            expected_output_tokens=300,
            total=1000,
        ),
        triggers=["new_quiz_request"],
        outputs_to=["Score Tracker"],
        eval_criteria=EvalCriteria(
            good_output_examples=["Well-formed MCQ", "Clear explanation"],
            bad_output_examples=["Ambiguous options", "Wrong answer marked"],
            automated_eval_suggestions=[
                "Check JSON schema",
                "Verify answer in options",
                "Check difficulty alignment",
            ],
            human_review_criteria=["Medical accuracy"],
        ),
    )


@pytest.fixture
def sample_critic_feedback() -> CriticFeedback:
    return CriticFeedback(
        module_name="Question Generator",
        overall_score=7.5,
        passed=True,
        category_scores={
            "ambiguity": CategoryScore(score=8.0),
            "hallucination_risk": CategoryScore(score=7.0),
            "missing_constraints": CategoryScore(score=7.5),
            "edge_cases": CategoryScore(score=7.0),
            "token_efficiency": CategoryScore(score=8.0),
        },
        issues=[],
        iteration=1,
        summary="Solid prompt with good structure. Minor improvements possible.",
    )


@pytest.fixture
def mock_llm_response():
    """Factory fixture to create mock LLM responses."""
    def _make_response(data: dict) -> MagicMock:
        mock = MagicMock()
        mock.content = json.dumps(data)
        return mock
    return _make_response
