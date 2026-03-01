"""Integration tests for checkpoint save/load with real file I/O."""

import pytest

from prompter.config import Settings
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
from prompter.utils.checkpoint import load_checkpoint, save_checkpoint


# ── Helpers ───────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    defaults = {"groq_api_key": "test-key", "groq_model": "llama-3.3-70b-versatile"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _make_module_map() -> ModuleMap:
    return ModuleMap(
        project_name="Test Project",
        domain_classification=DomainClassification(primary="Education"),
        interaction_model=InteractionType.adaptive,
        interaction_model_rationale="Test rationale.",
        modules=[
            Module(
                name="Question Generator",
                description="Generates questions.",
                requires_ai=True,
                ai_justification="NLG.",
                ai_capability_needed="generation",
                interaction_type=InteractionType.batch,
                data_inputs=["topic"],
                data_outputs=["question"],
                failure_modes=["Off-topic"],
            ),
        ],
        module_count=1,
        ai_module_count=1,
    )


def _make_artifact(module_name: str = "Question Generator") -> PromptArtifact:
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
                written_by=["Question Generator"], read_by=["Feedback"],
                updated_on="generated", default="",
            ),
        },
        handoff_conditions=[
            HandoffCondition(
                from_agent="Question Generator", to_agent="Feedback",
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


def _make_feedback(module_name: str, score: float = 8.5) -> CriticFeedback:
    return CriticFeedback(
        module_name=module_name,
        overall_score=score,
        passed=True,
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


def _make_full_state() -> dict:
    """Create a fully-populated pipeline state for testing."""
    settings = _make_settings()
    state = create_initial_state(
        project_idea="a medical quizzing platform",
        config=settings.safe_dict(),
        run_id="test-checkpoint-001",
    )
    state["module_map"] = _make_module_map()
    state["prompt_artifacts"] = [_make_artifact("Question Generator")]
    state["inter_agent_map"] = _make_inter_agent_map()
    state["critic_feedback"] = [[_make_feedback("Question Generator")]]
    state["current_iteration"] = 1
    state["all_passed"] = True
    state["agent_durations"] = {"analyzer": 1.5, "architect": 3.2}
    state["token_usage"] = {"analyzer": 500, "architect": 1200}
    state["last_checkpoint"] = "critique"
    return state


# ── Tests ─────────────────────────────────────────────────────────────


class TestCheckpointSaveLoad:
    """Tests for checkpoint serialization and deserialization with real files."""

    def test_save_creates_directory_and_file(self, tmp_path, monkeypatch):
        """save_checkpoint creates the run directory and pipeline_state.json."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={"model": "test"}, run_id="run-001",
        )
        path = save_checkpoint(state, "run-001")
        assert path.exists()
        assert path.name == "pipeline_state.json"
        assert path.parent.name == "run-001"

    def test_save_returns_path(self, tmp_path, monkeypatch):
        """save_checkpoint returns a Path object to the saved file."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-002",
        )
        path = save_checkpoint(state, "run-002")
        assert path == tmp_path / "run-002" / "pipeline_state.json"

    def test_load_from_file_path(self, tmp_path, monkeypatch):
        """load_checkpoint loads correctly from a direct file path."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test idea", config={}, run_id="run-003",
        )
        path = save_checkpoint(state, "run-003")
        loaded = load_checkpoint(path)
        assert loaded["project_idea"] == "test idea"

    def test_load_from_directory_path(self, tmp_path, monkeypatch):
        """load_checkpoint resolves directory to pipeline_state.json inside."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-004",
        )
        save_checkpoint(state, "run-004")
        loaded = load_checkpoint(tmp_path / "run-004")
        assert loaded["project_idea"] == "test"

    def test_load_nonexistent_raises_file_not_found(self):
        """load_checkpoint raises FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent/path/to/checkpoint")

    def test_roundtrip_minimal_state(self, tmp_path, monkeypatch):
        """Minimal state round-trips with all scalar fields matching."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="round trip test", config={"key": "val"},
            run_id="run-005", max_iterations=5, quality_threshold=8.0,
        )
        path = save_checkpoint(state, "run-005")
        loaded = load_checkpoint(path)

        assert loaded["project_idea"] == "round trip test"
        assert loaded["config"] == {"key": "val"}
        assert loaded["run_id"] == "run-005"
        assert loaded["max_iterations"] == 5
        assert loaded["quality_threshold"] == 8.0
        assert loaded["current_iteration"] == 0
        assert loaded["all_passed"] is False

    def test_roundtrip_module_map(self, tmp_path, monkeypatch):
        """ModuleMap Pydantic model survives save/load as a ModuleMap instance."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-006",
        )
        state["module_map"] = _make_module_map()
        path = save_checkpoint(state, "run-006")
        loaded = load_checkpoint(path)

        assert isinstance(loaded["module_map"], ModuleMap)
        assert loaded["module_map"].project_name == "Test Project"
        assert loaded["module_map"].module_count == 1

    def test_roundtrip_prompt_artifacts(self, tmp_path, monkeypatch):
        """list[PromptArtifact] round-trips correctly."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-007",
        )
        state["prompt_artifacts"] = [
            _make_artifact("Question Generator"),
            _make_artifact("Feedback Generator"),
        ]
        path = save_checkpoint(state, "run-007")
        loaded = load_checkpoint(path)

        assert len(loaded["prompt_artifacts"]) == 2
        assert all(isinstance(a, PromptArtifact) for a in loaded["prompt_artifacts"])
        assert loaded["prompt_artifacts"][0].module_name == "Question Generator"
        assert loaded["prompt_artifacts"][1].module_name == "Feedback Generator"

    def test_roundtrip_nested_critic_feedback(self, tmp_path, monkeypatch):
        """list[list[CriticFeedback]] survives serialization round-trip."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-008",
        )
        state["critic_feedback"] = [
            [_make_feedback("Question Generator", 7.5)],
            [_make_feedback("Question Generator", 8.5)],
        ]
        path = save_checkpoint(state, "run-008")
        loaded = load_checkpoint(path)

        assert len(loaded["critic_feedback"]) == 2
        assert len(loaded["critic_feedback"][0]) == 1
        assert len(loaded["critic_feedback"][1]) == 1
        assert isinstance(loaded["critic_feedback"][0][0], CriticFeedback)
        assert loaded["critic_feedback"][0][0].overall_score == 7.5
        assert loaded["critic_feedback"][1][0].overall_score == 8.5

    def test_roundtrip_full_pipeline_state(self, tmp_path, monkeypatch):
        """Fully populated state round-trips with all Pydantic models restored."""
        monkeypatch.setattr("prompter.utils.checkpoint._STATE_DIR", tmp_path)
        state = _make_full_state()
        path = save_checkpoint(state, "full-001")
        loaded = load_checkpoint(path)

        # Pydantic models restored to correct types
        assert isinstance(loaded["module_map"], ModuleMap)
        assert all(isinstance(a, PromptArtifact) for a in loaded["prompt_artifacts"])
        assert isinstance(loaded["inter_agent_map"], InterAgentMap)
        for iteration in loaded["critic_feedback"]:
            for fb in iteration:
                assert isinstance(fb, CriticFeedback)

        # Scalar fields match
        assert loaded["project_idea"] == "a medical quizzing platform"
        assert loaded["current_iteration"] == 1
        assert loaded["all_passed"] is True
        assert loaded["last_checkpoint"] == "critique"
        assert loaded["agent_durations"]["analyzer"] == 1.5
        assert loaded["token_usage"]["architect"] == 1200
