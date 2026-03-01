"""Unit tests for all Pydantic data models."""

import pytest
from pydantic import ValidationError

from prompter.models.module_map import (
    DomainClassification,
    InteractionType,
    Module,
    ModuleMap,
)
from prompter.models.prompt_artifact import (
    ChainCondition,
    ContextSlot,
    EvalCriteria,
    PromptArtifact,
    PromptChain,
    TokenEstimate,
)
from prompter.models.inter_agent_map import (
    ContextPollutionRule,
    HandoffCondition,
    InterAgentMap,
    SharedMemoryField,
    Trigger,
)
from prompter.models.critic_feedback import (
    CategoryScore,
    CriticFeedback,
    Issue,
    IssueCategory,
    Severity,
)
from prompter.models.final_output import (
    FinalOutputArtifact,
    PipelineMetadata,
)


# --- ModuleMap tests ---

class TestDomainClassification:
    def test_valid(self):
        dc = DomainClassification(primary="Education", secondary=["Healthcare"])
        assert dc.primary == "Education"
        assert dc.secondary == ["Healthcare"]

    def test_defaults(self):
        dc = DomainClassification(primary="Tech")
        assert dc.secondary == []

    def test_missing_primary_fails(self):
        with pytest.raises(ValidationError):
            DomainClassification()


class TestModule:
    def test_valid_ai_module(self, sample_module):
        assert sample_module.requires_ai is True
        assert sample_module.ai_justification is not None

    def test_valid_non_ai_module(self):
        m = Module(
            name="Logger",
            description="Logs events.",
            requires_ai=False,
            interaction_type=InteractionType.batch,
            data_inputs=["event"],
            data_outputs=["log_entry"],
            failure_modes=["Disk full"],
        )
        assert m.ai_justification is None

    def test_invalid_interaction_type(self):
        with pytest.raises(ValidationError):
            Module(
                name="Test",
                description="Test",
                requires_ai=False,
                interaction_type="invalid_type",
                data_inputs=[],
                data_outputs=[],
                failure_modes=[],
            )


class TestModuleMap:
    def test_valid(self, sample_module_map):
        assert sample_module_map.module_count == 2
        assert sample_module_map.ai_module_count == 1

    def test_clarification_defaults(self):
        mm = ModuleMap(
            project_name="Test",
            domain_classification=DomainClassification(primary="Tech"),
            interaction_model=InteractionType.batch,
            interaction_model_rationale="Simple batch processing.",
            modules=[],
            module_count=0,
            ai_module_count=0,
        )
        assert mm.needs_clarification is False
        assert mm.clarification_questions == []

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ModuleMap(project_name="Test")


# --- PromptArtifact tests ---

class TestTokenEstimate:
    def test_valid(self):
        te = TokenEstimate(
            system_tokens=500,
            expected_context_tokens=200,
            expected_output_tokens=300,
            total=1000,
        )
        assert te.total == 1000

    def test_with_warning(self):
        te = TokenEstimate(
            system_tokens=5000,
            expected_context_tokens=2000,
            expected_output_tokens=3000,
            total=10000,
            budget_warning="Exceeds 8K threshold",
        )
        assert te.budget_warning is not None


class TestEvalCriteria:
    def test_valid(self):
        ec = EvalCriteria(
            good_output_examples=["Example 1", "Example 2"],
            bad_output_examples=["Bad 1", "Bad 2"],
            automated_eval_suggestions=["Check 1", "Check 2", "Check 3"],
            human_review_criteria=["Criteria 1"],
        )
        assert len(ec.good_output_examples) == 2

    def test_too_few_good_examples(self):
        with pytest.raises(ValidationError):
            EvalCriteria(
                good_output_examples=["Only one"],
                bad_output_examples=["Bad 1", "Bad 2"],
                automated_eval_suggestions=["Check 1", "Check 2", "Check 3"],
                human_review_criteria=["Criteria 1"],
            )

    def test_too_few_automated_suggestions(self):
        with pytest.raises(ValidationError):
            EvalCriteria(
                good_output_examples=["Good 1", "Good 2"],
                bad_output_examples=["Bad 1", "Bad 2"],
                automated_eval_suggestions=["Only two", "Suggestions"],
                human_review_criteria=["Criteria 1"],
            )

    def test_empty_human_review(self):
        with pytest.raises(ValidationError):
            EvalCriteria(
                good_output_examples=["Good 1", "Good 2"],
                bad_output_examples=["Bad 1", "Bad 2"],
                automated_eval_suggestions=["Check 1", "Check 2", "Check 3"],
                human_review_criteria=[],
            )


class TestPromptArtifact:
    def test_valid(self, sample_prompt_artifact):
        assert sample_prompt_artifact.module_name == "Question Generator"
        assert sample_prompt_artifact.secondary_technique is None
        assert sample_prompt_artifact.prompt_chain is None

    def test_with_chain(self, sample_prompt_artifact):
        chain = PromptChain(
            conditions=[
                ChainCondition(
                    condition="engagement_level == low",
                    next_prompt="Difficulty Adapter",
                    context_passed=["user_score", "topic"],
                )
            ]
        )
        data = sample_prompt_artifact.model_dump()
        data["prompt_chain"] = chain.model_dump()
        artifact = PromptArtifact.model_validate(data)
        assert artifact.prompt_chain is not None
        assert len(artifact.prompt_chain.conditions) == 1

    def test_serialization_roundtrip(self, sample_prompt_artifact):
        json_str = sample_prompt_artifact.model_dump_json()
        restored = PromptArtifact.model_validate_json(json_str)
        assert restored == sample_prompt_artifact


# --- InterAgentMap tests ---

class TestInterAgentMap:
    def test_valid(self):
        iam = InterAgentMap(
            shared_memory_schema={
                "current_topic": SharedMemoryField(
                    type="string",
                    description="Active quiz topic",
                    written_by=["Question Generator"],
                    read_by=["Score Tracker"],
                    updated_on="new_quiz_start",
                    default="",
                )
            },
            handoff_conditions=[
                HandoffCondition(
                    from_agent="Question Generator",
                    to_agent="Score Tracker",
                    condition="quiz_complete == true",
                    data_passed={"score": "Final quiz score"},
                    format="JSON object",
                    fallback_if_incomplete="Use partial score",
                )
            ],
            context_pollution_rules=[
                ContextPollutionRule(
                    protected_data="correct_answer",
                    excluded_agents=["Hint Provider"],
                    risk="Leaking answers defeats quiz purpose",
                    enforcement="scoping",
                )
            ],
            trigger_map=[
                Trigger(
                    event="quiz_complete",
                    activates=["Score Tracker"],
                    priority_order=["Score Tracker"],
                    execution="sequential",
                    error_fallback="Log error, skip scoring",
                )
            ],
        )
        assert len(iam.shared_memory_schema) == 1
        assert len(iam.handoff_conditions) == 1


# --- CriticFeedback tests ---

class TestCategoryScore:
    def test_valid(self):
        cs = CategoryScore(score=8.5)
        assert cs.score == 8.5

    def test_out_of_range_high(self):
        with pytest.raises(ValidationError):
            CategoryScore(score=11.0)

    def test_out_of_range_low(self):
        with pytest.raises(ValidationError):
            CategoryScore(score=-1.0)

    def test_boundary_values(self):
        assert CategoryScore(score=0.0).score == 0.0
        assert CategoryScore(score=10.0).score == 10.0


class TestIssue:
    def test_valid(self):
        issue = Issue(
            category=IssueCategory.ambiguity,
            severity=Severity.medium,
            location="You should generate questions",
            description="'Should' is ambiguous — use 'must'.",
            suggestion="Replace 'should' with 'must' for clarity.",
        )
        assert issue.category == IssueCategory.ambiguity


class TestCriticFeedback:
    def test_valid(self, sample_critic_feedback):
        assert sample_critic_feedback.passed is True
        assert sample_critic_feedback.overall_score == 7.5

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            CriticFeedback(
                module_name="Test",
                overall_score=15.0,
                passed=True,
                category_scores={},
                issues=[],
                iteration=1,
                summary="Invalid score",
            )

    def test_with_issues(self):
        issue = Issue(
            category=IssueCategory.missing_constraints,
            severity=Severity.high,
            location="Generate any question",
            description="No difficulty constraint",
            suggestion="Add difficulty level requirement",
        )
        feedback = CriticFeedback(
            module_name="Test",
            overall_score=5.0,
            passed=False,
            category_scores={
                "missing_constraints": CategoryScore(score=4.0, issues=[issue])
            },
            issues=[issue],
            iteration=1,
            summary="Needs work.",
        )
        assert not feedback.passed
        assert len(feedback.issues) == 1


# --- FinalOutputArtifact tests ---

class TestFinalOutputArtifact:
    def test_valid(self, sample_prompt_artifact):
        iam = InterAgentMap(
            shared_memory_schema={},
            handoff_conditions=[],
            context_pollution_rules=[],
            trigger_map=[],
        )
        metadata = PipelineMetadata(
            total_modules=2,
            ai_modules=1,
            total_estimated_tokens=1000,
            average_quality_score=7.5,
            critic_iterations_used=1,
            total_pipeline_tokens_consumed=5000,
            generation_duration_seconds=30.0,
        )
        foa = FinalOutputArtifact(
            project="Test Project",
            generated_at="2026-03-01T00:00:00Z",
            modules=[sample_prompt_artifact],
            inter_agent_map=iam,
            pipeline_metadata=metadata,
        )
        assert foa.version == "1.0"
        assert foa.model_target == "llama-3.3-70b-versatile"
        assert len(foa.modules) == 1

    def test_json_roundtrip(self, sample_prompt_artifact):
        iam = InterAgentMap(
            shared_memory_schema={},
            handoff_conditions=[],
            context_pollution_rules=[],
            trigger_map=[],
        )
        metadata = PipelineMetadata(
            total_modules=1,
            ai_modules=1,
            total_estimated_tokens=1000,
            average_quality_score=8.0,
            critic_iterations_used=1,
            total_pipeline_tokens_consumed=3000,
            generation_duration_seconds=15.0,
        )
        foa = FinalOutputArtifact(
            project="Test",
            generated_at="2026-03-01T00:00:00Z",
            modules=[sample_prompt_artifact],
            inter_agent_map=iam,
            pipeline_metadata=metadata,
        )
        json_str = foa.model_dump_json()
        restored = FinalOutputArtifact.model_validate_json(json_str)
        assert restored.project == foa.project
        assert len(restored.modules) == len(foa.modules)
