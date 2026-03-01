"""Refiner agent — revises PromptArtifacts that failed the Critic's quality threshold."""

import logging
import time

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.models.critic_feedback import CriticFeedback
from prompter.models.prompt_artifact import PromptArtifact
from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def _build_user_message(
    artifact: PromptArtifact,
    feedback: CriticFeedback,
    state: PipelineState,
) -> str:
    """Build the user message with original artifact, feedback, and module map."""
    parts = [
        "## Original PromptArtifact (to revise)\n\n"
        f"```json\n{artifact.model_dump_json(indent=2)}\n```\n\n"
        "## Critic Feedback (issues to address)\n\n"
        f"```json\n{feedback.model_dump_json(indent=2)}\n```\n"
    ]

    if state.get("module_map") is not None:
        parts.append(
            "\n## ModuleMap Context\n\n"
            f"```json\n{state['module_map'].model_dump_json(indent=2)}\n```\n"
        )

    return "".join(parts)


def refine(state: PipelineState, settings: Settings | None = None) -> dict:
    """Revise PromptArtifacts that failed the Critic's quality assessment.

    This is a LangGraph node function. It reads the latest `critic_feedback`,
    identifies failing artifacts, and produces revised versions.

    Args:
        state: Current pipeline state with `critic_feedback` populated.
        settings: Settings instance. If None, loads from env.

    Returns:
        Dict of state updates including revised `prompt_artifacts`,
        incremented `current_iteration`, telemetry, and checkpoint marker.
    """
    critic_feedback = state.get("critic_feedback", [])
    if not critic_feedback:
        raise ValueError("Cannot run Refiner: critic_feedback is empty. Run Critic first.")

    artifacts = state["prompt_artifacts"]
    if not artifacts:
        raise ValueError("Cannot run Refiner: prompt_artifacts is empty.")

    if settings is None:
        settings = Settings()

    # Get latest round of feedback
    latest_feedback = critic_feedback[-1]

    # Build a lookup from module_name to feedback
    feedback_by_module: dict[str, CriticFeedback] = {
        f.module_name: f for f in latest_feedback
    }

    # Identify which artifacts failed
    failed_modules = {f.module_name for f in latest_feedback if not f.passed}

    if not failed_modules:
        logger.info("Refiner: all artifacts passed — nothing to revise.")
        return {
            "prompt_artifacts": artifacts,
            "current_iteration": state.get("current_iteration", 0) + 1,
            "token_usage": dict(state.get("token_usage", {})),
            "agent_durations": dict(state.get("agent_durations", {})),
            "last_checkpoint": "refine",
        }

    logger.info(f"Refiner: revising {len(failed_modules)} failed artifacts")

    system_prompt = load_prompt("refiner_system")
    revised_artifacts: list[PromptArtifact] = []
    total_start = time.time()

    for artifact in artifacts:
        if artifact.module_name not in failed_modules:
            # Keep passing artifacts as-is
            revised_artifacts.append(artifact)
            continue

        feedback = feedback_by_module[artifact.module_name]
        logger.info(f"  Revising: {artifact.module_name} (score: {feedback.overall_score:.1f})")

        user_message = _build_user_message(artifact, feedback, state)

        revised: PromptArtifact = call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=PromptArtifact,
            temperature=settings.creative_temperature,
            max_tokens=8192,
            settings=settings,
        )

        logger.info(f"    Revised: {artifact.module_name}")
        revised_artifacts.append(revised)

    total_duration = time.time() - total_start
    logger.info(f"Refiner complete in {total_duration:.1f}s")

    # Merge telemetry
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["refiner"] = token_usage.get("refiner", 0)
    agent_durations["refiner"] = total_duration

    return {
        "prompt_artifacts": revised_artifacts,
        "current_iteration": state.get("current_iteration", 0) + 1,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "refine",
    }
