"""Communication Designer agent — generates InterAgentMap from ModuleMap and PromptArtifacts."""

import logging
import time

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.models.inter_agent_map import InterAgentMap
from prompter.models.module_map import ModuleMap
from prompter.models.prompt_artifact import PromptArtifact
from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def _build_user_message(module_map: ModuleMap, artifacts: list[PromptArtifact]) -> str:
    """Build the user message with ModuleMap and PromptArtifacts context."""
    artifacts_json = ",\n".join(a.model_dump_json(indent=2) for a in artifacts)
    return (
        f"## ModuleMap\n\n"
        f"```json\n{module_map.model_dump_json(indent=2)}\n```\n\n"
        f"## PromptArtifacts\n\n"
        f"```json\n[\n{artifacts_json}\n]\n```\n"
    )


def _validate_data_coverage(
    module_map: ModuleMap,
    inter_agent_map: InterAgentMap,
) -> list[str]:
    """Check that all data_inputs/data_outputs appear in shared memory or handoffs.

    Returns a list of warning strings for any uncovered data fields.
    """
    # Collect all shared memory field names
    shared_fields = set(inter_agent_map.shared_memory_schema.keys())

    # Collect all data fields mentioned in handoff data_passed
    handoff_fields: set[str] = set()
    for handoff in inter_agent_map.handoff_conditions:
        handoff_fields.update(handoff.data_passed.keys())

    covered = shared_fields | handoff_fields

    warnings: list[str] = []
    for module in module_map.modules:
        for field in module.data_inputs + module.data_outputs:
            # Normalize: check if the field name appears as a substring in any covered field
            # to handle cases like "topic" matching "current_topic"
            field_lower = field.lower().replace(" ", "_")
            found = any(
                field_lower in cf.lower().replace(" ", "_")
                or cf.lower().replace(" ", "_") in field_lower
                for cf in covered
            )
            if not found:
                warnings.append(
                    f"Module '{module.name}' field '{field}' not found in "
                    f"shared memory or handoff conditions"
                )

    return warnings


def design_communication(state: PipelineState, settings: Settings | None = None) -> dict:
    """Generate an InterAgentMap defining inter-module communication infrastructure.

    This is a LangGraph node function. It reads `module_map` and `prompt_artifacts`
    from state, and produces an InterAgentMap with shared memory, handoffs,
    pollution rules, and triggers.

    Args:
        state: Current pipeline state with `module_map` and `prompt_artifacts` populated.
        settings: Settings instance with API credentials. If None, loads from env.

    Returns:
        Dict of state updates including `inter_agent_map`, telemetry,
        and checkpoint marker.
    """
    module_map = state["module_map"]
    prompt_artifacts = state["prompt_artifacts"]

    if module_map is None:
        raise ValueError("Cannot run Communication Designer: module_map is None.")
    if not prompt_artifacts:
        raise ValueError("Cannot run Communication Designer: prompt_artifacts is empty.")

    if settings is None:
        settings = Settings()

    # Load system prompt
    system_prompt = load_prompt("communication_designer_system")

    # Build user message with full context
    user_message = _build_user_message(module_map, prompt_artifacts)

    logger.info("Communication Designer: generating inter-agent map...")
    start_time = time.time()

    inter_agent_map: InterAgentMap = call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=InterAgentMap,
        temperature=settings.creative_temperature,
        max_tokens=8192,
        settings=settings,
    )

    duration = time.time() - start_time

    # Post-processing: validate data coverage (FR-004.6)
    coverage_warnings = _validate_data_coverage(module_map, inter_agent_map)
    if coverage_warnings:
        for warning in coverage_warnings:
            logger.warning(f"  Data coverage gap: {warning}")
    else:
        logger.info("  All data fields covered in shared memory or handoffs.")

    logger.info(
        f"Communication Designer complete in {duration:.1f}s — "
        f"{len(inter_agent_map.shared_memory_schema)} shared fields, "
        f"{len(inter_agent_map.handoff_conditions)} handoffs, "
        f"{len(inter_agent_map.context_pollution_rules)} pollution rules, "
        f"{len(inter_agent_map.trigger_map)} triggers"
    )

    # Merge telemetry with existing state values
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["communication_designer"] = token_usage.get("communication_designer", 0)
    agent_durations["communication_designer"] = duration

    return {
        "inter_agent_map": inter_agent_map,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "design_communication",
    }
