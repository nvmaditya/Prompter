"""Integration tests for the LangGraph pipeline, routing functions, and CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prompter.agents.packager import NarrativeResponse
from prompter.cli import app
from prompter.config import Settings
from prompter.graph import (
    NODE_ORDER,
    build_graph,
    check_clarification_needed,
    get_next_node,
    should_continue_refining,
)
from prompter.models.critic_feedback import CategoryScore, CriticFeedback
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
from prompter.state import create_initial_state


# ── Shared Test Helpers ──────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    defaults = {"groq_api_key": "test-key", "groq_model": "llama-3.3-70b-versatile"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_module_map(needs_clarification: bool = False) -> ModuleMap:
    if needs_clarification:
        return ModuleMap(
            project_name="Unclear Project",
            domain_classification=DomainClassification(primary="Unknown"),
            interaction_model=InteractionType.batch,
            interaction_model_rationale="Unclear.",
            needs_clarification=True,
            clarification_questions=["What does this do?", "Who are the users?"],
            modules=[],
            module_count=0,
            ai_module_count=0,
        )
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
                data_outputs=["feedback"],
                failure_modes=["Generic"],
                depends_on=["Question Generator"],
            ),
        ],
        module_count=2,
        ai_module_count=2,
    )


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


def _make_feedback(module_name: str, score: float = 8.5, passed: bool = True) -> CriticFeedback:
    return CriticFeedback(
        module_name=module_name,
        overall_score=score,
        passed=passed,
        category_scores={
            "ambiguity": CategoryScore(score=score),
            "hallucination_risk": CategoryScore(score=score),
            "missing_constraints": CategoryScore(score=score),
            "edge_cases": CategoryScore(score=score),
            "token_efficiency": CategoryScore(score=score),
        },
        issues=[],
        iteration=1,
        summary=f"Assessment for {module_name}.",
    )


AI_MODULE_NAMES = ["Question Generator", "Feedback Generator"]


def _make_initial_state() -> dict:
    settings = _make_settings()
    return create_initial_state(
        project_idea="a quizzing platform for medical students",
        config=settings.safe_dict(),
        run_id="test-graph-001",
    )


# ── Graph Construction Tests ─────────────────────────────────────────


class TestGraphConstruction:
    """Tests for build_graph()."""

    def test_build_graph_returns_compiled_graph(self):
        """build_graph returns a compiled graph with invoke and stream."""
        settings = _make_settings()
        graph = build_graph(settings)
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_build_graph_with_all_valid_entry_nodes(self):
        """build_graph accepts all valid node names as entry_node."""
        settings = _make_settings()
        for node in NODE_ORDER:
            graph = build_graph(settings, entry_node=node)
            assert graph is not None

    def test_build_graph_rejects_invalid_entry_node(self):
        """build_graph raises ValueError for unknown entry_node."""
        settings = _make_settings()
        with pytest.raises(ValueError, match="Unknown entry node"):
            build_graph(settings, entry_node="nonexistent")


# ── Routing Function Tests ───────────────────────────────────────────


class TestRoutingFunctions:
    """Tests for the two conditional edge routing functions."""

    def test_clarification_needed_proceeds(self):
        """Returns 'proceed' when needs_clarification is False."""
        state = {"needs_clarification": False}
        assert check_clarification_needed(state) == "proceed"

    def test_clarification_needed_exits(self):
        """Returns 'needs_clarification' when True."""
        state = {"needs_clarification": True}
        assert check_clarification_needed(state) == "needs_clarification"

    def test_clarification_needed_defaults_to_proceed(self):
        """Returns 'proceed' when key is missing (defensive)."""
        assert check_clarification_needed({}) == "proceed"

    def test_should_continue_packages_when_all_passed(self):
        """Returns 'package' when all_passed is True."""
        state = {"all_passed": True, "current_iteration": 0, "max_iterations": 3}
        assert should_continue_refining(state) == "package"

    def test_should_continue_packages_at_max_iterations(self):
        """Returns 'package' when iterations exhausted."""
        state = {"all_passed": False, "current_iteration": 3, "max_iterations": 3}
        assert should_continue_refining(state) == "package"

    def test_should_continue_refines_when_below_threshold(self):
        """Returns 'refine' when not passed and iterations remain."""
        state = {"all_passed": False, "current_iteration": 1, "max_iterations": 3}
        assert should_continue_refining(state) == "refine"


# ── Resume Helper Tests ──────────────────────────────────────────────


class TestGetNextNode:
    """Tests for get_next_node() resume routing."""

    def test_after_analyze_goes_to_architect(self):
        state = {"needs_clarification": False}
        assert get_next_node("analyze", state) == "architect"

    def test_after_analyze_clarification_returns_none(self):
        state = {"needs_clarification": True}
        assert get_next_node("analyze", state) is None

    def test_after_architect_goes_to_design_communication(self):
        assert get_next_node("architect", {}) == "design_communication"

    def test_after_design_communication_goes_to_critique(self):
        assert get_next_node("design_communication", {}) == "critique"

    def test_after_critique_routes_to_refine(self):
        state = {"all_passed": False, "current_iteration": 0, "max_iterations": 3}
        assert get_next_node("critique", state) == "refine"

    def test_after_critique_routes_to_package(self):
        state = {"all_passed": True, "current_iteration": 1, "max_iterations": 3}
        assert get_next_node("critique", state) == "package"

    def test_after_refine_goes_to_critique(self):
        assert get_next_node("refine", {}) == "critique"

    def test_after_package_returns_none(self):
        assert get_next_node("package", {}) is None

    def test_unknown_checkpoint_raises(self):
        with pytest.raises(ValueError, match="Unknown last_checkpoint"):
            get_next_node("bogus", {})


# ── End-to-End Pipeline Tests ────────────────────────────────────────


class TestPipelineEndToEnd:
    """Full pipeline tests with all LLM calls mocked."""

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_happy_path_produces_final_output(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_packager, mock_json, mock_md, mock_scaffold,
    ):
        """Full pipeline runs end-to-end when all artifacts pass first time."""
        mock_analyze.return_value = _make_module_map()
        mock_architect.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]
        mock_comm.return_value = _make_inter_agent_map()
        mock_critic.side_effect = [_make_feedback(n) for n in AI_MODULE_NAMES]
        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        settings = _make_settings()
        graph = build_graph(settings)
        state = _make_initial_state()

        result = graph.invoke(state)

        assert result["last_checkpoint"] == "package"
        assert result["final_output"] is not None
        assert result["all_passed"] is True

    @patch("prompter.agents.analyzer.call_llm")
    def test_clarification_exits_early(self, mock_analyze):
        """Pipeline exits after analyze when clarification is needed."""
        mock_analyze.return_value = _make_module_map(needs_clarification=True)

        settings = _make_settings()
        graph = build_graph(settings)
        state = _make_initial_state()

        result = graph.invoke(state)

        assert result["needs_clarification"] is True
        assert result["last_checkpoint"] == "analyze"
        assert result.get("prompt_artifacts") == []

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.refiner.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_critic_loop_iterates(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_refiner, mock_packager, mock_json, mock_md, mock_scaffold,
    ):
        """Critic loop iterates: fail first time, pass after refine."""
        mock_analyze.return_value = _make_module_map()
        mock_architect.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]
        mock_comm.return_value = _make_inter_agent_map()

        # First critique: fail. Second critique: pass.
        fail_feedbacks = [_make_feedback(n, score=5.0, passed=False) for n in AI_MODULE_NAMES]
        pass_feedbacks = [_make_feedback(n, score=8.5, passed=True) for n in AI_MODULE_NAMES]
        mock_critic.side_effect = fail_feedbacks + pass_feedbacks

        # Refiner returns revised artifacts
        mock_refiner.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]

        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        settings = _make_settings()
        graph = build_graph(settings)
        state = _make_initial_state()

        result = graph.invoke(state)

        assert result["last_checkpoint"] == "package"
        assert result["all_passed"] is True
        # Should have 2 rounds of critic feedback
        assert len(result["critic_feedback"]) == 2

    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.refiner.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_max_iterations_stops_loop(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_refiner, mock_packager, mock_json, mock_md, mock_scaffold,
    ):
        """Pipeline packages output when max_iterations reached even if not all pass."""
        mock_analyze.return_value = _make_module_map()
        mock_architect.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]
        mock_comm.return_value = _make_inter_agent_map()

        # With max_iterations=2, the loop runs:
        # critique(fail) → refine → critique(fail) → refine → critique(fail) → package
        # = 3 critique rounds (6 feedbacks) + 2 refine rounds (4 artifacts)
        fail_feedbacks = [_make_feedback(n, score=5.0, passed=False) for n in AI_MODULE_NAMES]
        mock_critic.side_effect = fail_feedbacks * 3  # 3 rounds
        mock_refiner.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES] * 2  # 2 rounds

        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")

        settings = _make_settings()
        graph = build_graph(settings)
        state = _make_initial_state()
        state["max_iterations"] = 2

        result = graph.invoke(state)

        assert result["last_checkpoint"] == "package"
        assert result["final_output"] is not None
        assert result["current_iteration"] >= 2


# ── CLI Command Tests ────────────────────────────────────────────────


class TestCLICommands:
    """Tests for the CLI commands integrated with the graph."""

    @patch("prompter.utils.checkpoint.save_checkpoint")
    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_generate_full_pipeline(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_packager, mock_json, mock_md, mock_scaffold, mock_checkpoint,
    ):
        """CLI generate command runs the full pipeline."""
        mock_analyze.return_value = _make_module_map()
        mock_architect.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]
        mock_comm.return_value = _make_inter_agent_map()
        mock_critic.side_effect = [_make_feedback(n) for n in AI_MODULE_NAMES]
        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")
        mock_checkpoint.return_value = Path(".prompter_state/test/pipeline_state.json")

        runner = CliRunner()
        result = runner.invoke(app, ["generate", "a quizzing platform for medical students"])

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Pipeline complete" in result.output

    @patch("prompter.utils.checkpoint.save_checkpoint")
    @patch("prompter.agents.analyzer.call_llm")
    def test_interactive_pauses_after_analyzer(self, mock_analyze, mock_checkpoint):
        """Interactive mode shows module map and prompts, user declines."""
        mock_analyze.return_value = _make_module_map()
        mock_checkpoint.return_value = Path(".prompter_state/test/pipeline_state.json")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["interactive", "a quizzing platform for medical students"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Medical Quiz Platform" in result.output

    @patch("prompter.utils.checkpoint.save_checkpoint")
    @patch("prompter.agents.packager.write_scaffolding")
    @patch("prompter.agents.packager.write_markdown")
    @patch("prompter.agents.packager.write_json")
    @patch("prompter.agents.packager.call_llm")
    @patch("prompter.agents.critic.call_llm")
    @patch("prompter.agents.communication_designer.call_llm")
    @patch("prompter.agents.architect.call_llm")
    @patch("prompter.agents.analyzer.call_llm")
    def test_interactive_continues_after_approval(
        self, mock_analyze, mock_architect, mock_comm, mock_critic,
        mock_packager, mock_json, mock_md, mock_scaffold, mock_checkpoint,
    ):
        """Interactive mode runs full pipeline when user approves."""
        mock_analyze.return_value = _make_module_map()
        mock_architect.side_effect = [_make_artifact(n) for n in AI_MODULE_NAMES]
        mock_comm.return_value = _make_inter_agent_map()
        mock_critic.side_effect = [_make_feedback(n) for n in AI_MODULE_NAMES]
        mock_packager.return_value = NarrativeResponse(narrative="Summary.")
        mock_json.return_value = Path("output/prompt_config.json")
        mock_md.return_value = Path("output/architecture_spec.md")
        mock_scaffold.return_value = Path("output/scaffolding")
        mock_checkpoint.return_value = Path(".prompter_state/test/pipeline_state.json")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["interactive", "a quizzing platform for medical students"],
            input="y\n",
        )

        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
        assert "Pipeline complete" in result.output
