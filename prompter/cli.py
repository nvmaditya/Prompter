"""CLI interface for the Prompt Engineering Engine."""

import re
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompter import __version__

app = typer.Typer(
    name="prompter",
    help="A multi-agent AI system that builds prompt architectures from project ideas.",
    no_args_is_help=True,
)
console = Console(stderr=True)
output_console = Console()


def version_callback(value: bool) -> None:
    if value:
        output_console.print(f"prompter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True
    ),
) -> None:
    """Prompter — build prompt architectures from project ideas."""


def _load_idea(idea: str) -> str:
    """Load idea from argument — supports inline text or file path."""
    path = Path(idea)
    if path.exists() and path.suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            console.print("[red]Error: Input file is empty.[/red]")
            raise typer.Exit(code=1)
        return text
    return idea


def _validate_idea(idea: str) -> None:
    """Validate idea length constraints (10-10,000 chars)."""
    if len(idea) < 10:
        console.print(
            "[red]Error: Project idea is too short (minimum 10 characters).[/red]\n"
            "Provide a more detailed description of your project."
        )
        raise typer.Exit(code=1)
    if len(idea) > 10_000:
        console.print(
            "[red]Error: Project idea is too long (maximum 10,000 characters).[/red]\n"
            "Summarize your project idea to fit within the limit."
        )
        raise typer.Exit(code=1)


def _slugify(text: str, max_length: int = 50) -> str:
    """Convert text to a filesystem-safe slug.

    Lowercase, replace non-alphanumeric runs with hyphens, truncate on word boundary.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug or "project"


def _display_module_map(module_map) -> None:
    """Display a ModuleMap using rich formatting."""
    output_console.print(Panel(
        f"[bold]{module_map.project_name}[/bold]\n"
        f"Domain: {module_map.domain_classification.primary}"
        + (f" + {', '.join(module_map.domain_classification.secondary)}" if module_map.domain_classification.secondary else "")
        + f"\nInteraction Model: {module_map.interaction_model.value}\n"
        f"Rationale: {module_map.interaction_model_rationale}",
        title="Project Analysis",
        border_style="green",
    ))

    table = Table(title=f"Modules ({module_map.module_count} total, {module_map.ai_module_count} AI)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Module", style="cyan", min_width=20)
    table.add_column("AI?", justify="center", width=5)
    table.add_column("Type", width=14)
    table.add_column("Description", min_width=30)

    for i, mod in enumerate(module_map.modules, 1):
        ai_marker = "[green]Yes[/green]" if mod.requires_ai else "[dim]No[/dim]"
        table.add_row(str(i), mod.name, ai_marker, mod.interaction_type.value, mod.description)

    output_console.print(table)


# ── Node name → telemetry key mapping ────────────────────────────────

_DURATION_KEY_MAP = {
    "analyze": "analyzer",
    "architect": "architect",
    "design_communication": "communication_designer",
    "critique": "critic",
    "refine": "refiner",
    "package": "packager",
}

_NODE_LABELS = {
    "analyze": "Analyzing project idea",
    "architect": "Generating prompt artifacts",
    "design_communication": "Designing inter-agent communication",
    "critique": "Evaluating prompt quality",
    "refine": "Refining prompts",
    "package": "Packaging output",
}


def _display_critic_scores(state: dict) -> None:
    """Display critic scores after a critique node completes."""
    critic_feedback = state.get("critic_feedback", [])
    if not critic_feedback:
        return

    latest = critic_feedback[-1]
    for fb in latest:
        status = "[green]PASS[/green]" if fb.passed else "[red]REFINING...[/red]"
        output_console.print(
            f"    Module: {fb.module_name:<30s} "
            f"{fb.overall_score:.1f}/10  {status}"
        )


def _display_completion(state: dict, settings) -> None:
    """Display final summary after successful pipeline completion."""
    final = state.get("final_output")
    if final is None:
        console.print("[yellow]Pipeline completed but no final output was produced.[/yellow]")
        return

    total_duration = sum(state.get("agent_durations", {}).values())
    output_console.print(Panel(
        f"[bold green]Pipeline complete![/bold green]\n\n"
        f"  Output directory: [cyan]{settings.output_dir}[/cyan]\n"
        f"  Total modules: {final.pipeline_metadata.total_modules}\n"
        f"  AI modules: {final.pipeline_metadata.ai_modules}\n"
        f"  Average quality: {final.pipeline_metadata.average_quality_score:.1f}/10\n"
        f"  Critic iterations: {final.pipeline_metadata.critic_iterations_used}\n"
        f"  Total duration: {total_duration:.1f}s",
        title="Results",
        border_style="green",
    ))


def _run_pipeline(graph, state: dict, run_id: str, verbose: bool) -> dict:
    """Stream graph execution, checkpoint after each node, display progress.

    Args:
        graph: Compiled LangGraph StateGraph.
        state: Initial (or resumed) PipelineState dict.
        run_id: Pipeline run identifier for checkpointing.
        verbose: Whether to show debug output.

    Returns:
        The final pipeline state dict.
    """
    from prompter.utils.checkpoint import save_checkpoint

    try:
        for event in graph.stream(state, stream_mode="updates"):
            # event is {"node_name": {updated_state_keys...}}
            node_name = list(event.keys())[0]
            updates = event[node_name]
            state.update(updates)

            # Save checkpoint after each node
            save_checkpoint(state, run_id)

            # Display node completion
            duration_key = _DURATION_KEY_MAP.get(node_name, node_name)
            duration = state.get("agent_durations", {}).get(duration_key, 0)
            label = _NODE_LABELS.get(node_name, node_name)
            console.print(
                f"  [green]\u2713[/green] {label}  "
                f"[dim]({duration:.1f}s)[/dim]"
            )

            # After critique: display score table
            if node_name == "critique":
                _display_critic_scores(state)

    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        if verbose:
            console.print_exception()
        # Save checkpoint so user can resume
        try:
            save_checkpoint(state, run_id)
            console.print(
                f"[yellow]Checkpoint saved. Resume with: "
                f"--resume .prompter_state/{run_id}[/yellow]"
            )
        except Exception:
            pass
        raise typer.Exit(code=1)

    return state


@app.command()
def generate(
    idea: str = typer.Argument(..., help="Project idea description (text or path to .txt file)."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory path."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint path."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable detailed debug output."),
) -> None:
    """Generate a complete prompt architecture from a project idea."""
    from prompter.config import Settings
    from prompter.graph import build_graph, get_next_node
    from prompter.state import create_initial_state
    from prompter.utils.checkpoint import load_checkpoint
    from prompter.utils.logging import setup_logging

    setup_logging(verbose)

    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        console.print(
            f"[red]Configuration error: {e}[/red]\n\n"
            "Make sure GROQ_API_KEY is set in your environment or .env file.\n"
            "See .env.example for the required configuration."
        )
        raise typer.Exit(code=3)

    # Apply CLI overrides
    if output_dir:
        settings = settings.model_copy(update={"output_dir": output_dir})
    if verbose:
        settings = settings.model_copy(update={"verbose": verbose})

    # ── Resume path ──────────────────────────────────────────────────
    if resume:
        try:
            state = load_checkpoint(resume)
        except FileNotFoundError:
            console.print(f"[red]Checkpoint not found: {resume}[/red]")
            raise typer.Exit(code=1)

        last_cp = state.get("last_checkpoint", "")
        next_node = get_next_node(last_cp, state)

        if next_node is None:
            console.print("[yellow]Pipeline already complete. Nothing to resume.[/yellow]")
            raise typer.Exit(code=0)

        run_id = state.get("run_id", "resumed")
        console.print(
            f"[cyan]Resuming from checkpoint: {last_cp} \u2192 {next_node}[/cyan]"
        )

        graph = build_graph(settings, entry_node=next_node)
        state = _run_pipeline(graph, state, run_id, verbose)

        if state.get("needs_clarification", False):
            output_console.print(Panel(
                "\n".join(
                    f"  {i}. {q}"
                    for i, q in enumerate(state["clarification_questions"], 1)
                ),
                title="[yellow]Clarification Needed[/yellow]",
                border_style="yellow",
            ))
            raise typer.Exit(code=2)

        _display_completion(state, settings)
        return

    # ── New execution path ───────────────────────────────────────────
    idea_text = _load_idea(idea)
    _validate_idea(idea_text)
    slug = _slugify(idea_text)

    # Auto-derive output subfolder from task name (unless user specified --output)
    if not output_dir:
        settings = settings.model_copy(update={"output_dir": str(Path(settings.output_dir) / slug)})

    run_id = f"{slug}_{uuid.uuid4().hex[:12]}"
    state = create_initial_state(
        project_idea=idea_text,
        config=settings.safe_dict(),
        run_id=run_id,
        max_iterations=settings.max_iterations,
        quality_threshold=settings.quality_threshold,
    )

    console.print(Panel(
        f"Run ID: [bold]{run_id}[/bold]",
        title="Prompter Pipeline",
        border_style="cyan",
    ))

    graph = build_graph(settings)
    state = _run_pipeline(graph, state, run_id, verbose)

    # Handle clarification exit
    if state.get("needs_clarification", False):
        output_console.print(Panel(
            "\n".join(
                f"  {i}. {q}"
                for i, q in enumerate(state["clarification_questions"], 1)
            ),
            title="[yellow]Clarification Needed[/yellow]",
            border_style="yellow",
        ))
        output_console.print(
            "\n[yellow]The idea needs more detail. "
            "Please refine your description and try again.[/yellow]"
        )
        raise typer.Exit(code=2)

    _display_completion(state, settings)


@app.command()
def interactive(
    idea: str = typer.Argument(..., help="Project idea description (text or path to .txt file)."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory path."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable detailed debug output."),
) -> None:
    """Interactively generate a prompt architecture with a review gate after analysis."""
    from prompter.agents.analyzer import analyze
    from prompter.config import Settings
    from prompter.graph import build_graph
    from prompter.state import create_initial_state
    from prompter.utils.checkpoint import save_checkpoint
    from prompter.utils.logging import setup_logging

    setup_logging(verbose)

    idea_text = _load_idea(idea)
    _validate_idea(idea_text)
    slug = _slugify(idea_text)

    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(code=3)

    if output_dir:
        settings = settings.model_copy(update={"output_dir": output_dir})
    else:
        settings = settings.model_copy(update={"output_dir": str(Path(settings.output_dir) / slug)})
    if verbose:
        settings = settings.model_copy(update={"verbose": verbose})

    run_id = f"{slug}_{uuid.uuid4().hex[:12]}"
    state = create_initial_state(
        project_idea=idea_text,
        config=settings.safe_dict(),
        run_id=run_id,
        max_iterations=settings.max_iterations,
        quality_threshold=settings.quality_threshold,
    )

    # Step 1: Run analyzer directly (not through graph)
    with console.status("[bold green]Analyzing project idea...", spinner="dots"):
        try:
            updates = analyze(state, settings=settings)
        except Exception as e:
            console.print(f"[red]Analyzer failed: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1)

    state.update(updates)
    save_checkpoint(state, run_id)

    # Handle clarification
    if state["needs_clarification"]:
        output_console.print(Panel(
            "\n".join(
                f"  {i}. {q}"
                for i, q in enumerate(state["clarification_questions"], 1)
            ),
            title="[yellow]Clarification Needed[/yellow]",
            border_style="yellow",
        ))
        raise typer.Exit(code=2)

    # Step 2: Display results and prompt for approval
    _display_module_map(state["module_map"])

    duration = state["agent_durations"].get("analyzer", 0)
    console.print(f"\n[dim]Analyzer completed in {duration:.1f}s[/dim]")

    proceed = typer.confirm("\nProceed with prompt generation?", default=True)
    if not proceed:
        console.print("[yellow]Aborted by user.[/yellow]")
        raise typer.Exit(code=0)

    # Step 3: Build graph from architect onward and run
    console.print()
    graph = build_graph(settings, entry_node="architect")
    state = _run_pipeline(graph, state, run_id, verbose)

    _display_completion(state, settings)
