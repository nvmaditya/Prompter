"""Markdown specification writer — hybrid LLM narrative + Python-generated tables."""

import logging
from pathlib import Path

from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def _build_module_summary_table(state: PipelineState) -> str:
    """Build a Markdown table summarizing modules, techniques, and scores."""
    artifacts = state.get("prompt_artifacts", [])
    latest_feedback = {}
    if state.get("critic_feedback"):
        for fb in state["critic_feedback"][-1]:
            latest_feedback[fb.module_name] = fb

    rows = []
    for artifact in artifacts:
        fb = latest_feedback.get(artifact.module_name)
        score = f"{fb.overall_score:.1f}" if fb else "N/A"
        status = ""
        if fb:
            if fb.passed:
                status = "PASS"
            else:
                status = "**FAIL** (below threshold)"
        technique = artifact.primary_technique
        if artifact.secondary_technique:
            technique += f" + {artifact.secondary_technique}"
        rows.append(f"| {artifact.module_name} | {technique} | {score} | {status} |")

    header = "| Module | Technique(s) | Score | Status |\n| --- | --- | --- | --- |"
    return header + "\n" + "\n".join(rows)


def _build_token_budget_table(state: PipelineState) -> str:
    """Build a Markdown table of per-module token estimates."""
    artifacts = state.get("prompt_artifacts", [])
    rows = []
    total = 0
    for artifact in artifacts:
        te = artifact.token_estimate
        warning = ""
        if te.budget_warning:
            warning = f" ({te.budget_warning})"
        rows.append(
            f"| {artifact.module_name} | {te.system_tokens} | "
            f"{te.expected_context_tokens} | {te.expected_output_tokens} | "
            f"{te.total}{warning} |"
        )
        total += te.total

    header = (
        "| Module | System | Context | Output | Total |\n"
        "| --- | ---: | ---: | ---: | ---: |"
    )
    footer = f"| **Total** | | | | **{total}** |"
    return header + "\n" + "\n".join(rows) + "\n" + footer


def _build_communication_overview(state: PipelineState) -> str:
    """Build communication design overview from InterAgentMap."""
    iam = state.get("inter_agent_map")
    if iam is None:
        return "*Communication design not available.*"

    sections = []

    # Shared Memory
    sections.append("### Shared Memory Fields\n")
    for name, field in iam.shared_memory_schema.items():
        sections.append(
            f"- **`{name}`** ({field.type}): {field.description}\n"
            f"  - Written by: {', '.join(field.written_by)}\n"
            f"  - Read by: {', '.join(field.read_by)}\n"
            f"  - Default: `{field.default}`"
        )

    # Handoffs
    sections.append("\n### Handoff Conditions\n")
    for h in iam.handoff_conditions:
        sections.append(
            f"- **{h.from_agent}** -> **{h.to_agent}**\n"
            f"  - Condition: `{h.condition}`\n"
            f"  - Fallback: {h.fallback_if_incomplete}"
        )

    # Triggers
    sections.append("\n### Trigger Map\n")
    for t in iam.trigger_map:
        sections.append(
            f"- **{t.event}**\n"
            f"  - Activates: {', '.join(t.activates)} ({t.execution})\n"
            f"  - Error fallback: {t.error_fallback}"
        )

    return "\n".join(sections)


def _build_prompt_details(state: PipelineState) -> str:
    """Build per-module prompt details with system prompt in code blocks."""
    artifacts = state.get("prompt_artifacts", [])
    sections = []
    for artifact in artifacts:
        sections.append(f"### {artifact.module_name}\n")
        sections.append(f"**Role**: {artifact.agent_role}\n")
        sections.append(f"**Technique**: {artifact.primary_technique}")
        if artifact.secondary_technique:
            sections.append(f" + {artifact.secondary_technique}")
        sections.append(f"\n\n**Rationale**: {artifact.technique_rationale}\n")
        sections.append(f"#### System Prompt\n\n```\n{artifact.system_prompt}\n```\n")

        # Context slots
        if artifact.context_slots:
            sections.append("#### Context Slots\n")
            for slot in artifact.context_slots:
                req = "Required" if slot.required else "Optional"
                sections.append(
                    f"- `{{{{{{{slot.variable}}}}}}}` ({req}): {slot.description}\n"
                    f"  - Source: {slot.source} | Fallback: {slot.fallback}"
                )
            sections.append("")

    return "\n".join(sections)


def write_markdown(
    state: PipelineState,
    narrative: str,
    output_dir: Path,
) -> Path:
    """Write the architecture specification as a Markdown document.

    Combines LLM-generated narrative sections with Python-generated
    structured tables and code blocks.

    Args:
        state: Full pipeline state with all fields populated.
        narrative: LLM-generated executive summary and module descriptions.
        output_dir: Directory to write the Markdown file into.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "architecture_spec.md"

    project_name = "Unknown Project"
    if state.get("module_map"):
        project_name = state["module_map"].project_name

    sections = [
        f"# {project_name} — Prompt Architecture Specification\n",
        "## Table of Contents\n",
        "1. [Executive Summary](#executive-summary)",
        "2. [Module Overview](#module-overview)",
        "3. [Module Details](#module-details)",
        "4. [Token Budget](#token-budget)",
        "5. [Inter-Agent Communication](#inter-agent-communication)",
        "6. [Quality Assessment](#quality-assessment)",
        "7. [Implementation Notes](#implementation-notes)\n",
        "---\n",
        "## Executive Summary\n",
        narrative + "\n",
        "---\n",
        "## Module Overview\n",
        _build_module_summary_table(state) + "\n",
        "---\n",
        "## Module Details\n",
        _build_prompt_details(state) + "\n",
        "---\n",
        "## Token Budget\n",
        _build_token_budget_table(state) + "\n",
        "---\n",
        "## Inter-Agent Communication\n",
        _build_communication_overview(state) + "\n",
        "---\n",
        "## Quality Assessment\n",
    ]

    # Quality details from latest critic feedback
    if state.get("critic_feedback") and state["critic_feedback"]:
        latest = state["critic_feedback"][-1]
        for fb in latest:
            status = "PASS" if fb.passed else "**FAIL**"
            sections.append(f"### {fb.module_name}: {fb.overall_score:.1f}/10 {status}\n")
            sections.append(f"{fb.summary}\n")
            if fb.issues:
                sections.append("**Issues:**\n")
                for issue in fb.issues:
                    sections.append(
                        f"- [{issue.severity.value.upper()}] {issue.category.value}: "
                        f"{issue.description}"
                    )
                sections.append("")
    else:
        sections.append("*No quality assessment data available.*\n")

    sections.append("---\n")
    sections.append("## Implementation Notes\n")
    sections.append(
        "This architecture was generated by the Prompt Engineering Engine. "
        "See `prompt_config.json` for the machine-readable configuration and "
        "`scaffolding/` for starter code.\n"
    )

    content = "\n".join(sections)
    file_path.write_text(content, encoding="utf-8")

    logger.info(f"Markdown spec written to {file_path} ({len(content)} bytes)")
    return file_path
