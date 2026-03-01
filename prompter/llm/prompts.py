"""Prompt template loader — reads system prompts from the prompt_templates directory."""

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent.parent / "prompt_templates"


def load_prompt(template_name: str) -> str:
    """Load a prompt template by name (without extension).

    Args:
        template_name: Name of the template file (e.g., 'analyzer_system').

    Returns:
        The template text content.

    Raises:
        FileNotFoundError: If template file doesn't exist.
    """
    path = _TEMPLATE_DIR / f"{template_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
