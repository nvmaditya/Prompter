"""Analyzer agent — decomposes project ideas into structured module maps."""

import logging
import time

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.models.module_map import ModuleMap
from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def analyze(state: PipelineState, settings: Settings | None = None) -> dict:
    """Analyze a project idea and produce a ModuleMap.

    This is a LangGraph node function. It reads `project_idea` from state,
    calls the LLM with the analyzer system prompt, and returns state updates.

    Args:
        state: Current pipeline state with `project_idea` populated.
        settings: Settings instance with API credentials. If None, loads from env.

    Returns:
        Dict of state updates including `module_map`, clarification fields,
        telemetry, and checkpoint marker.
    """
    project_idea = state["project_idea"]

    if settings is None:
        settings = Settings()

    system_prompt = load_prompt("analyzer_system")

    logger.info("Analyzing project idea...")
    start_time = time.time()

    module_map: ModuleMap = call_llm(
        system_prompt=system_prompt,
        user_message=project_idea,
        response_model=ModuleMap,
        temperature=settings.creative_temperature,
        max_tokens=4096,
        settings=settings,
    )

    duration = time.time() - start_time
    logger.info(f"Analysis complete in {duration:.1f}s — {module_map.module_count} modules identified")

    # Merge telemetry with existing state values
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["analyzer"] = token_usage.get("analyzer", 0)
    agent_durations["analyzer"] = duration

    return {
        "module_map": module_map,
        "needs_clarification": module_map.needs_clarification,
        "clarification_questions": module_map.clarification_questions,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "analyze",
    }
