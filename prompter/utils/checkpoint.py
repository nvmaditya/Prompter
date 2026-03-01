"""Pipeline checkpoint save/load for resume support."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path(".prompter_state")


def _serialize_state(state: dict) -> dict:
    """Convert PipelineState to a JSON-serializable dict.

    Pydantic models within the state are converted via model_dump().
    """
    from pydantic import BaseModel

    result = {}
    for key, value in state.items():
        if isinstance(value, BaseModel):
            result[key] = {"__pydantic__": type(value).__name__, "data": value.model_dump()}
        elif isinstance(value, list):
            result[key] = [
                {"__pydantic__": type(v).__name__, "data": v.model_dump()}
                if isinstance(v, BaseModel)
                else (
                    [
                        {"__pydantic__": type(inner).__name__, "data": inner.model_dump()}
                        if isinstance(inner, BaseModel)
                        else inner
                        for inner in v
                    ]
                    if isinstance(v, list)
                    else v
                )
                for v in value
            ]
        elif isinstance(value, dict):
            serialized_dict = {}
            for k, v in value.items():
                if isinstance(v, BaseModel):
                    serialized_dict[k] = {
                        "__pydantic__": type(v).__name__,
                        "data": v.model_dump(),
                    }
                elif isinstance(v, dict) and "artifact" in v:
                    # BestVersion TypedDict
                    artifact = v["artifact"]
                    serialized_dict[k] = {
                        "artifact": artifact if isinstance(artifact, dict) else artifact,
                        "score": v["score"],
                    }
                else:
                    serialized_dict[k] = v
            result[key] = serialized_dict
        else:
            result[key] = value
    return result


def save_checkpoint(state: dict, run_id: str) -> Path:
    """Save pipeline state to a checkpoint file.

    Args:
        state: The PipelineState dict to save.
        run_id: Unique identifier for this pipeline run.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = _STATE_DIR / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pipeline_state.json"

    serialized = _serialize_state(state)
    checkpoint_path.write_text(json.dumps(serialized, indent=2, default=str), encoding="utf-8")
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict:
    """Load pipeline state from a checkpoint file.

    Args:
        path: Path to the checkpoint directory or JSON file.

    Returns:
        The deserialized state dict.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
    """
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "pipeline_state.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    logger.info(f"Checkpoint loaded: {checkpoint_path}")
    return data
