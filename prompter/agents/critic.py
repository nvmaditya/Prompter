"""Critic agent — evaluates PromptArtifacts for quality across 5 dimensions."""

import logging
import time

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.models.critic_feedback import CriticFeedback
from prompter.models.prompt_artifact import PromptArtifact
from prompter.state import BestVersion, PipelineState

logger = logging.getLogger(__name__)


def _build_user_message(
    artifact: PromptArtifact,
    state: PipelineState,
) -> str:
    """Build the user message with artifact, module map, and inter-agent map context."""
    parts = [
        "## PromptArtifact to Evaluate\n\n"
        f"```json\n{artifact.model_dump_json(indent=2)}\n```\n"
    ]

    if state.get("module_map") is not None:
        parts.append(
            "\n## ModuleMap Context\n\n"
            f"```json\n{state['module_map'].model_dump_json(indent=2)}\n```\n"
        )

    if state.get("inter_agent_map") is not None:
        parts.append(
            "\n## InterAgentMap Context\n\n"
            f"```json\n{state['inter_agent_map'].model_dump_json(indent=2)}\n```\n"
        )

    return "".join(parts)


def critique(state: PipelineState, settings: Settings | None = None) -> dict:
    """Evaluate each PromptArtifact for quality and assign scores.

    This is a LangGraph node function. It reads `prompt_artifacts` from state,
    evaluates each one against the quality rubric, and produces CriticFeedback.

    Args:
        state: Current pipeline state with `prompt_artifacts` populated.
        settings: Settings instance. If None, loads from env.

    Returns:
        Dict of state updates including `critic_feedback`, `all_passed`,
        `best_prompt_versions`, telemetry, and checkpoint marker.
    """
    artifacts = state["prompt_artifacts"]
    if not artifacts:
        raise ValueError("Cannot run Critic: prompt_artifacts is empty.")

    if settings is None:
        settings = Settings()

    quality_threshold = state.get("quality_threshold", settings.quality_threshold)
    current_iteration = state.get("current_iteration", 0) + 1

    # Load system prompt and inject runtime values
    raw_prompt = load_prompt("critic_system")
    system_prompt = raw_prompt.replace(
        "{{quality_threshold}}", str(quality_threshold)
    ).replace(
        "{{iteration}}", str(current_iteration)
    )

    logger.info(f"Critic: evaluating {len(artifacts)} artifacts (iteration {current_iteration})")
    feedbacks: list[CriticFeedback] = []
    total_start = time.time()

    for idx, artifact in enumerate(artifacts):
        logger.info(f"  [{idx + 1}/{len(artifacts)}] Evaluating: {artifact.module_name}")

        user_message = _build_user_message(artifact, state)

        feedback: CriticFeedback = call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=CriticFeedback,
            temperature=settings.analytical_temperature,
            max_tokens=4096,
            settings=settings,
        )

        status = "PASS" if feedback.passed else "FAIL"
        logger.info(f"    {artifact.module_name}: {feedback.overall_score:.1f}/10 [{status}]")
        feedbacks.append(feedback)

    total_duration = time.time() - total_start

    # Determine if all passed
    all_passed = all(f.passed for f in feedbacks)
    logger.info(
        f"Critic complete in {total_duration:.1f}s — "
        f"{'ALL PASSED' if all_passed else 'SOME FAILED'}"
    )

    # Update best prompt versions
    best_versions = dict(state.get("best_prompt_versions", {}))
    for artifact, feedback in zip(artifacts, feedbacks):
        module_name = artifact.module_name
        current_best = best_versions.get(module_name)
        if current_best is None or feedback.overall_score > current_best["score"]:
            best_versions[module_name] = BestVersion(
                artifact=artifact.model_dump(),
                score=feedback.overall_score,
            )

    # Append this iteration's feedback to the outer list
    existing_feedback = list(state.get("critic_feedback", []))
    existing_feedback.append(feedbacks)

    # Merge telemetry
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["critic"] = token_usage.get("critic", 0)
    agent_durations["critic"] = total_duration

    return {
        "critic_feedback": existing_feedback,
        "all_passed": all_passed,
        "best_prompt_versions": best_versions,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "critique",
    }
