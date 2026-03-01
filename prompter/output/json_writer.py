"""JSON output writer — deterministic FinalOutputArtifact serialization."""

import logging
from pathlib import Path

from prompter.models.final_output import FinalOutputArtifact

logger = logging.getLogger(__name__)


def write_json(artifact: FinalOutputArtifact, output_dir: Path) -> Path:
    """Write FinalOutputArtifact as prompt_config.json.

    This is a purely deterministic operation — no LLM involved.
    The prompt text is serialized exactly as-is from the Pydantic model (FR-007.7).

    Args:
        artifact: The complete pipeline output artifact.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "prompt_config.json"

    json_content = artifact.model_dump_json(indent=2)
    file_path.write_text(json_content, encoding="utf-8")

    logger.info(f"JSON config written to {file_path} ({len(json_content)} bytes)")
    return file_path
