"""Prompt engineering technique registry — extensible catalog for the Architect agent."""

TECHNIQUE_REGISTRY: dict[str, dict[str, str]] = {
    "chain_of_thought": {
        "name": "Chain-of-Thought",
        "when_to_use": "Reasoning-heavy modules (feedback, explanation, analysis)",
        "prompt_pattern": "Think step by step before producing your final answer.",
    },
    "few_shot": {
        "name": "Few-Shot",
        "when_to_use": "Output format consistency (question gen, scoring, structured data)",
        "prompt_pattern": "Here are examples of correct input-output pairs: ...",
    },
    "role_prompting": {
        "name": "Role Prompting",
        "when_to_use": "Persona-driven modules (tutor, reviewer, assistant)",
        "prompt_pattern": "You are a [role]. Your expertise is in [domain].",
    },
    "output_constraints": {
        "name": "Output Constraints",
        "when_to_use": "Structured data output (JSON, specific formats)",
        "prompt_pattern": "You MUST respond in the following format: ...",
    },
    "socratic": {
        "name": "Socratic Technique",
        "when_to_use": "Learning and discovery flows (quizzing, guided exploration)",
        "prompt_pattern": "Guide the user through questions rather than giving answers.",
    },
}


def get_technique_catalog() -> str:
    """Format the technique registry as a readable catalog for injection into prompts."""
    lines = ["Available Prompt Engineering Techniques:\n"]
    for key, tech in TECHNIQUE_REGISTRY.items():
        lines.append(f"- **{tech['name']}** (key: `{key}`)")
        lines.append(f"  When to use: {tech['when_to_use']}")
        lines.append(f"  Pattern: {tech['prompt_pattern']}")
        lines.append("")
    return "\n".join(lines)
