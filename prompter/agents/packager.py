"""Packager agent — assembles final output and writes all deliverables."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from prompter.config import Settings
from prompter.llm.client import call_llm
from prompter.llm.prompts import load_prompt
from prompter.models.final_output import FinalOutputArtifact, PipelineMetadata
from prompter.models.prompt_artifact import PromptArtifact
from prompter.output.json_writer import write_json
from prompter.output.markdown_writer import write_markdown
from prompter.output.scaffold_writer import write_scaffolding
from prompter.state import PipelineState

logger = logging.getLogger(__name__)


class NarrativeResponse(BaseModel):
    """Thin wrapper so call_llm can parse the narrative via JSON."""
    narrative: str


def _resolve_best_artifacts(state: PipelineState) -> list[PromptArtifact]:
    """Select the best available artifacts.

    If best_prompt_versions is populated (from Critic iterations),
    use those over the current prompt_artifacts for modules that
    have a higher-scoring version stored.
    """
    artifacts = list(state["prompt_artifacts"])
    best_versions = state.get("best_prompt_versions", {})

    if not best_versions:
        return artifacts

    # Build lookup by module name
    result = []
    for artifact in artifacts:
        best = best_versions.get(artifact.module_name)
        if best is not None:
            # Restore the best-scoring artifact from its serialized form
            restored = PromptArtifact.model_validate(best["artifact"])
            result.append(restored)
        else:
            result.append(artifact)
    return result


def _compute_metadata(state: PipelineState, artifacts: list[PromptArtifact]) -> PipelineMetadata:
    """Compute pipeline metadata from state and final artifacts."""
    module_map = state.get("module_map")
    total_modules = module_map.module_count if module_map else len(artifacts)
    ai_modules = module_map.ai_module_count if module_map else len(artifacts)

    total_estimated_tokens = sum(a.token_estimate.total for a in artifacts)

    # Average quality score from latest critic feedback
    average_score = 0.0
    critic_feedback = state.get("critic_feedback", [])
    if critic_feedback:
        latest = critic_feedback[-1]
        if latest:
            average_score = sum(f.overall_score for f in latest) / len(latest)

    iterations_used = len(critic_feedback)

    total_pipeline_tokens = sum(state.get("token_usage", {}).values())
    total_duration = sum(state.get("agent_durations", {}).values())

    return PipelineMetadata(
        total_modules=total_modules,
        ai_modules=ai_modules,
        total_estimated_tokens=total_estimated_tokens,
        average_quality_score=round(average_score, 2),
        critic_iterations_used=iterations_used,
        total_pipeline_tokens_consumed=total_pipeline_tokens,
        generation_duration_seconds=round(total_duration, 2),
    )


def _build_narrative_user_message(state: PipelineState, artifacts: list[PromptArtifact]) -> str:
    """Build the user message for narrative generation."""
    parts = []

    # ModuleMap context
    module_map = state.get("module_map")
    if module_map:
        parts.append(
            "## ModuleMap\n\n"
            f"```json\n{module_map.model_dump_json(indent=2)}\n```\n"
        )

    # Artifacts summary (not full prompts — just key fields)
    parts.append("## PromptArtifacts Summary\n")
    for artifact in artifacts:
        parts.append(
            f"### {artifact.module_name}\n"
            f"- Role: {artifact.agent_role}\n"
            f"- Technique: {artifact.primary_technique}"
        )
        if artifact.secondary_technique:
            parts.append(f" + {artifact.secondary_technique}")
        parts.append(
            f"\n- Rationale: {artifact.technique_rationale}\n"
            f"- Token estimate: {artifact.token_estimate.total}\n"
        )

    # Quality scores
    critic_feedback = state.get("critic_feedback", [])
    if critic_feedback:
        latest = critic_feedback[-1]
        parts.append("## Quality Scores\n")
        for fb in latest:
            status = "PASS" if fb.passed else "FAIL"
            parts.append(f"- {fb.module_name}: {fb.overall_score:.1f}/10 ({status})")
            if fb.issues:
                for issue in fb.issues:
                    parts.append(f"  - [{issue.severity.value}] {issue.category.value}: {issue.description}")
        parts.append("")

    # Inter-agent map summary
    iam = state.get("inter_agent_map")
    if iam:
        parts.append(
            "## InterAgentMap\n\n"
            f"```json\n{iam.model_dump_json(indent=2)}\n```\n"
        )

    return "\n".join(parts)


def package(state: PipelineState, settings: Settings | None = None) -> dict:
    """Assemble final output, generate narrative, and write all deliverables.

    This is a LangGraph node function. It:
    1. Resolves the best artifacts (from best_prompt_versions or current)
    2. Assembles FinalOutputArtifact with PipelineMetadata
    3. Calls LLM for Markdown narrative
    4. Writes JSON, Markdown, and scaffolding output

    Args:
        state: Current pipeline state with all fields populated.
        settings: Settings instance. If None, loads from env.

    Returns:
        Dict of state updates including final_output, telemetry, and checkpoint.
    """
    if settings is None:
        settings = Settings()

    total_start = time.time()

    # Step 1: Resolve best artifacts
    artifacts = _resolve_best_artifacts(state)
    if not artifacts:
        raise ValueError("Cannot package: no prompt_artifacts available.")

    inter_agent_map = state.get("inter_agent_map")
    if inter_agent_map is None:
        raise ValueError("Cannot package: inter_agent_map is None.")

    # Step 2: Compute metadata
    metadata = _compute_metadata(state, artifacts)

    # Step 3: Assemble FinalOutputArtifact
    project_name = "Unknown Project"
    module_map = state.get("module_map")
    if module_map:
        project_name = module_map.project_name

    artifact = FinalOutputArtifact(
        project=project_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_target=settings.groq_model,
        modules=artifacts,
        inter_agent_map=inter_agent_map,
        pipeline_metadata=metadata,
    )

    # Step 4: Call LLM for narrative
    system_prompt = load_prompt("packager_system")
    user_message = _build_narrative_user_message(state, artifacts)

    narrative_response: NarrativeResponse = call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=NarrativeResponse,
        temperature=settings.packager_temperature,
        max_tokens=16384,
        settings=settings,
    )
    narrative = narrative_response.narrative

    # Step 5: Write outputs
    output_dir = Path(settings.output_dir)

    # Always write JSON
    json_path = write_json(artifact, output_dir)
    logger.info(f"JSON written: {json_path}")

    # Write Markdown
    md_path = write_markdown(state, narrative, output_dir)
    logger.info(f"Markdown written: {md_path}")

    # Write scaffolding if enabled
    if settings.scaffold_enabled:
        scaffold_path = write_scaffolding(state, output_dir)
        logger.info(f"Scaffolding written: {scaffold_path}")

    total_duration = time.time() - total_start
    logger.info(f"Packager complete in {total_duration:.1f}s")

    # Merge telemetry
    token_usage = dict(state.get("token_usage", {}))
    agent_durations = dict(state.get("agent_durations", {}))
    token_usage["packager"] = token_usage.get("packager", 0)
    agent_durations["packager"] = total_duration

    return {
        "final_output": artifact,
        "token_usage": token_usage,
        "agent_durations": agent_durations,
        "last_checkpoint": "package",
    }
