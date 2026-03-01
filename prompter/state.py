"""Pipeline state definition — the central TypedDict flowing through LangGraph."""

from typing import Optional, TypedDict

from prompter.models.critic_feedback import CriticFeedback
from prompter.models.final_output import FinalOutputArtifact
from prompter.models.inter_agent_map import InterAgentMap
from prompter.models.module_map import ModuleMap
from prompter.models.prompt_artifact import PromptArtifact


class BestVersion(TypedDict):
    artifact: dict  # Serialized PromptArtifact
    score: float


class PipelineState(TypedDict):
    # --- Input ---
    project_idea: str
    config: dict  # Serialized Settings (API key excluded)

    # --- Analyzer output ---
    module_map: Optional[ModuleMap]

    # --- Architect output (one per AI module) ---
    prompt_artifacts: list[PromptArtifact]

    # --- Communication Designer output ---
    inter_agent_map: Optional[InterAgentMap]

    # --- Critic output (outer = iterations, inner = per-module feedback) ---
    critic_feedback: list[list[CriticFeedback]]

    # --- Refiner tracking ---
    current_iteration: int
    max_iterations: int
    quality_threshold: float
    all_passed: bool

    # --- Best versions (module_name -> best artifact + score) ---
    best_prompt_versions: dict[str, BestVersion]

    # --- Final output ---
    final_output: Optional[FinalOutputArtifact]

    # --- Telemetry ---
    token_usage: dict[str, int]  # agent_name -> tokens_consumed
    agent_durations: dict[str, float]  # agent_name -> seconds

    # --- Pipeline control ---
    needs_clarification: bool
    clarification_questions: list[str]

    # --- Error recovery ---
    last_checkpoint: str  # Name of last successfully completed node
    run_id: str


def create_initial_state(
    project_idea: str,
    config: dict,
    run_id: str,
    max_iterations: int = 3,
    quality_threshold: float = 7.0,
) -> PipelineState:
    """Create a fresh PipelineState with defaults for a new pipeline run."""
    return PipelineState(
        project_idea=project_idea,
        config=config,
        module_map=None,
        prompt_artifacts=[],
        inter_agent_map=None,
        critic_feedback=[],
        current_iteration=0,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        all_passed=False,
        best_prompt_versions={},
        final_output=None,
        token_usage={},
        agent_durations={},
        needs_clarification=False,
        clarification_questions=[],
        last_checkpoint="",
        run_id=run_id,
    )
