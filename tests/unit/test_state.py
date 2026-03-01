"""Unit tests for PipelineState creation and defaults."""

from prompter.state import PipelineState, create_initial_state


class TestCreateInitialState:
    """Tests for the create_initial_state() factory function."""

    def test_returns_dict_with_all_expected_keys(self):
        """State dict contains all PipelineState TypedDict keys."""
        state = create_initial_state(
            project_idea="test idea for state",
            config={"groq_model": "llama-3.3-70b-versatile"},
            run_id="test-001",
        )
        expected_keys = {
            "project_idea", "config", "module_map", "prompt_artifacts",
            "inter_agent_map", "critic_feedback", "current_iteration",
            "max_iterations", "quality_threshold", "all_passed",
            "best_prompt_versions", "final_output", "token_usage",
            "agent_durations", "needs_clarification", "clarification_questions",
            "last_checkpoint", "run_id",
        }
        assert set(state.keys()) == expected_keys

    def test_project_idea_stored(self):
        """project_idea matches the input string."""
        state = create_initial_state(
            project_idea="a quizzing platform",
            config={}, run_id="r1",
        )
        assert state["project_idea"] == "a quizzing platform"

    def test_config_stored(self):
        """config matches the input dict."""
        cfg = {"groq_model": "llama-3.3-70b-versatile", "verbose": True}
        state = create_initial_state(
            project_idea="test", config=cfg, run_id="r1",
        )
        assert state["config"] == cfg

    def test_run_id_stored(self):
        """run_id matches the input."""
        state = create_initial_state(
            project_idea="test", config={}, run_id="run-abc-123",
        )
        assert state["run_id"] == "run-abc-123"

    def test_default_max_iterations(self):
        """Default max_iterations is 3."""
        state = create_initial_state(
            project_idea="test", config={}, run_id="r1",
        )
        assert state["max_iterations"] == 3

    def test_default_quality_threshold(self):
        """Default quality_threshold is 7.0."""
        state = create_initial_state(
            project_idea="test", config={}, run_id="r1",
        )
        assert state["quality_threshold"] == 7.0

    def test_custom_overrides(self):
        """Custom max_iterations and quality_threshold override defaults."""
        state = create_initial_state(
            project_idea="test", config={}, run_id="r1",
            max_iterations=5, quality_threshold=8.0,
        )
        assert state["max_iterations"] == 5
        assert state["quality_threshold"] == 8.0

    def test_empty_collections_initialized(self):
        """All collection fields start empty; Optionals start as None; bools start False."""
        state = create_initial_state(
            project_idea="test", config={}, run_id="r1",
        )
        # Lists empty
        assert state["prompt_artifacts"] == []
        assert state["critic_feedback"] == []
        assert state["clarification_questions"] == []

        # Dicts empty
        assert state["token_usage"] == {}
        assert state["agent_durations"] == {}
        assert state["best_prompt_versions"] == {}

        # Optionals None
        assert state["module_map"] is None
        assert state["inter_agent_map"] is None
        assert state["final_output"] is None

        # Bools False
        assert state["needs_clarification"] is False
        assert state["all_passed"] is False

        # Counters zero / empty
        assert state["current_iteration"] == 0
        assert state["last_checkpoint"] == ""
