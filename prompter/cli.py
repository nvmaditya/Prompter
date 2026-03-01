"""CLI interface for the Prompt Engineering Engine."""

from typing import Optional

import typer
from rich.console import Console

from prompter import __version__

app = typer.Typer(
    name="prompter",
    help="A multi-agent AI system that builds prompt architectures from project ideas.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"prompter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True
    ),
) -> None:
    """Prompter — build prompt architectures from project ideas."""


@app.command()
def generate(
    idea: str = typer.Argument(..., help="Project idea description (text or path to .txt file)."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory path."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint path."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable detailed debug output."),
) -> None:
    """Generate a complete prompt architecture from a project idea."""
    console.print("[yellow]generate command not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@app.command()
def interactive(
    idea: str = typer.Argument(..., help="Project idea description (text or path to .txt file)."),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory path."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable detailed debug output."),
) -> None:
    """Interactively generate a prompt architecture with review gates at each stage."""
    console.print("[yellow]interactive command not yet implemented[/yellow]")
    raise typer.Exit(code=0)
