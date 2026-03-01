"""Architect agent — generates PromptArtifacts for each AI module in the ModuleMap."""

import logging
import time

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.llm.techniques import get_technique_catalog
from prompter.models.module_map import ModuleMap
from prompter.models.prompt_artifact import PromptArtifact
from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def _build_user_message(module_map: ModuleMap, module_index: int) -> str:
    """Build the user message with full ModuleMap context and target module."""
    module = module_map.modules[module_index]
    return (
        f"## Full ModuleMap (for system context)\n\n"
        f"```json\n{module_map.model_dump_json(indent=2)}\n```\n\n"
        f"## Target Module to Generate PromptArtifact For\n\n"
        f"```json\n{module.model_dump_json(indent=2)}\n```\n"
    )


def architect(state: PipelineState, settings: Settings | None = None) -> dict:
    """Generate PromptArtifacts for all AI-requiring modules in the ModuleMap.

    This is a LangGraph node function. It reads `module_map` from state,
    iterates over AI modules, and produces a PromptArtifact per module.

    Args:
        state: Current pipeline state with `module_map` populated.
        settings: Settings instance with API credentials. If None, loads from env.

    Returns:
        Dict of state updates including `prompt_artifacts`, telemetry,
        and checkpoint marker.
    """
    module_map = state["module_map"]
    if module_map is None:
        raise ValueError("Cannot run Architect: module_map is None. Run Analyzer first.")

    if settings is None:
        settings = Settings()

    # Load and prepare system prompt with technique catalog injected
    raw_prompt = load_prompt("architect_system")
    technique_catalog = get_technique_catalog()
    system_prompt = raw_prompt.replace("{{TECHNIQUE_CATALOG}}", technique_catalog)

    # Identify AI modules
    ai_modules = [
        (i, m) for i, m in enumerate(module_map.modules) if m.requires_ai
    ]

    logger.info(f"Architect: generating prompts for {len(ai_modules)} AI modules")
    artifacts: list[PromptArtifact] = []
    total_start = time.time()

    for idx, (module_index, module) in enumerate(ai_modules):
        logger.info(f"  [{idx + 1}/{len(ai_modules)}] Generating artifact for: {module.name}")
        module_start = time.time()

        user_message = _build_user_message(module_map, module_index)

        artifact: PromptArtifact = call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=PromptArtifact,
            temperature=settings.creative_temperature,
            max_tokens=8192,
            settings=settings,
        )

        module_duration = time.time() - module_start
        logger.info(f"    Done: {module.name} ({module_duration:.1f}s)")
        artifacts.append(artifact)

    total_duration = time.time() - total_start
    logger.info(f"Architect complete: {len(artifacts)} artifacts in {total_duration:.1f}s")

    # Merge telemetry with existing state values
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["architect"] = token_usage.get("architect", 0)
    agent_durations["architect"] = total_duration

    return {
        "prompt_artifacts": artifacts,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "architect",
    }
