"""Output writers — JSON, Markdown, and scaffolding generation."""

from prompter.output.json_writer import write_json
from prompter.output.markdown_writer import write_markdown
from prompter.output.scaffold_writer import write_scaffolding

__all__ = ["write_json", "write_markdown", "write_scaffolding"]
