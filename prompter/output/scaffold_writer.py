"""Scaffold writer — generates starter code directory from pipeline state."""

import logging
import re
from pathlib import Path

from prompter.state import PipelineState

logger = logging.getLogger(__name__)


def _to_snake_case(name: str) -> str:
    """Convert a module name like 'Question Generator' to 'question_generator'."""
    # Replace non-alphanumeric chars with spaces, collapse, then underscorify
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", name).strip()
    return re.sub(r"\s+", "_", cleaned).lower()


def _build_readme(project_name: str, module_names: list[str]) -> str:
    """Build a README.md for the scaffold directory."""
    modules_list = "\n".join(f"- {name}" for name in module_names)
    return f"""# {project_name} — Scaffold

Auto-generated starter code for the **{project_name}** prompt architecture.

## Modules

{modules_list}

## Setup

1. Install dependencies:
   ```bash
   pip install openai  # or your preferred LLM client
   ```

2. Set your API key:
   ```bash
   export LLM_API_KEY="your-key-here"
   ```

3. Run the main entry point:
   ```bash
   python main.py
   ```

## Structure

```
scaffolding/
  prompts/          # System prompt text files (one per AI module)
  agents/           # Agent stubs with prompt loading logic
  config.py         # Configuration template
  main.py           # Entry point
  README.md         # This file
```

## Notes

- Prompt files in `prompts/` contain the system prompts exactly as generated.
- Agent stubs in `agents/` show how to load and use each prompt.
- Edit `config.py` to set your LLM provider and model.
- Context slot placeholders use `{{variable}}` syntax — replace at runtime.
"""


def _build_config_template(project_name: str) -> str:
    """Build a config.py template."""
    return f'''"""Configuration for {project_name}."""

# LLM Provider settings
LLM_PROVIDER = "openai"  # Change to your provider
LLM_MODEL = "gpt-4"  # Change to your target model
LLM_API_KEY = ""  # Set via environment variable LLM_API_KEY

# Pipeline settings
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60
'''


def _build_main_template(project_name: str, module_names: list[str]) -> str:
    """Build a main.py entry point template."""
    imports = []
    calls = []
    for name in module_names:
        snake = _to_snake_case(name)
        imports.append(f"from agents.{snake}_agent import run_{snake}")
        calls.append(f'    print("Running {name}...")')
        calls.append(f"    result = run_{snake}(context)")
        calls.append(f'    context["{snake}_output"] = result')
        calls.append("")

    imports_str = "\n".join(imports)
    calls_str = "\n".join(calls)

    return f'''"""{project_name} — Entry Point."""

{imports_str}


def main():
    """Run the {project_name} pipeline."""
    context = {{}}

{calls_str}
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
'''


def _build_agent_stub(module_name: str) -> str:
    """Build an agent stub Python file for one module."""
    snake = _to_snake_case(module_name)
    return f'''"""{module_name} agent stub."""

from pathlib import Path


def _load_prompt() -> str:
    """Load the system prompt for {module_name}."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "{snake}.txt"
    return prompt_path.read_text(encoding="utf-8")


def run_{snake}(context: dict) -> str:
    """Run the {module_name} agent.

    Args:
        context: Shared pipeline context dictionary.

    Returns:
        The agent's response text.
    """
    system_prompt = _load_prompt()

    # TODO: Replace with your LLM client call
    # Example:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {{"role": "system", "content": system_prompt}},
    #         {{"role": "user", "content": str(context)}},
    #     ],
    # )
    # return response.choices[0].message.content

    raise NotImplementedError("Replace this with your LLM client call.")
'''


def write_scaffolding(state: PipelineState, output_dir: Path) -> Path:
    """Write starter code scaffolding from pipeline state.

    Creates a directory structure with prompt files, agent stubs,
    configuration template, and entry point.

    Args:
        state: Full pipeline state with prompt_artifacts populated.
        output_dir: Parent directory (scaffolding/ will be created inside).

    Returns:
        Path to the scaffolding directory.
    """
    artifacts = state.get("prompt_artifacts", [])
    if not artifacts:
        raise ValueError("Cannot write scaffolding: no prompt_artifacts in state.")

    project_name = "Unknown Project"
    if state.get("module_map"):
        project_name = state["module_map"].project_name

    scaffold_dir = output_dir / "scaffolding"
    prompts_dir = scaffold_dir / "prompts"
    agents_dir = scaffold_dir / "agents"

    # Create directories
    prompts_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    ai_module_names = [a.module_name for a in artifacts]

    # Write prompt files (one per AI module)
    for artifact in artifacts:
        snake = _to_snake_case(artifact.module_name)
        prompt_path = prompts_dir / f"{snake}.txt"
        prompt_path.write_text(artifact.system_prompt, encoding="utf-8")

    # Write agent stubs
    for artifact in artifacts:
        snake = _to_snake_case(artifact.module_name)
        agent_path = agents_dir / f"{snake}_agent.py"
        agent_path.write_text(_build_agent_stub(artifact.module_name), encoding="utf-8")

    # Write agents/__init__.py
    init_path = agents_dir / "__init__.py"
    init_path.write_text("", encoding="utf-8")

    # Write README
    readme_path = scaffold_dir / "README.md"
    readme_path.write_text(
        _build_readme(project_name, ai_module_names), encoding="utf-8"
    )

    # Write config.py
    config_path = scaffold_dir / "config.py"
    config_path.write_text(_build_config_template(project_name), encoding="utf-8")

    # Write main.py
    main_path = scaffold_dir / "main.py"
    main_path.write_text(
        _build_main_template(project_name, ai_module_names), encoding="utf-8"
    )

    file_count = len(artifacts) * 2 + 4  # prompts + agents + README + config + main + __init__
    logger.info(f"Scaffolding written to {scaffold_dir} ({file_count} files)")
    return scaffold_dir
