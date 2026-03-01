# System Architecture Document: Prompt Engineering Engine

**Version**: 1.0
**Date**: 2026-03-01
**Status**: Draft
**References**: [PRD.md](./PRD.md) | [SRS.md](./SRS.md)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [LangGraph State Graph Design](#2-langgraph-state-graph-design)
3. [Agent Design](#3-agent-design)
4. [State Management](#4-state-management)
5. [Error Handling and Resilience](#5-error-handling-and-resilience)
6. [Project Source Code Structure](#6-project-source-code-structure)
7. [Configuration Management](#7-configuration-management)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment and Packaging](#9-deployment-and-packaging)
10. [Key Architectural Decisions](#10-key-architectural-decisions)

---

## 1. System Overview

### 1.1 High-Level Architecture

The Prompt Engineering Engine is a single-process, locally-executed CLI application with one external dependency: the Groq API for LLM inference. All orchestration, validation, and output generation runs on the user's machine.

```
User Input (CLI)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                   CLI Layer (Typer)                   │
│  prompter generate / interactive                     │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph StateGraph Engine              │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐    │
│  │ Analyzer │──▶│Architect │──▶│  Comm Designer │    │
│  └──────────┘   └──────────┘   └───────┬───────┘    │
│       │                                 │            │
│  (clarification                        ▼            │
│   halt if needed)          ┌──────────────────┐     │
│                            │     Critic       │     │
│                            └────────┬─────────┘     │
│                                     │               │
│                            ┌────────▼─────────┐     │
│                            │  score < thresh? │     │
│                            └──┬───────────┬───┘     │
│                          yes  │           │  no     │
│                     ┌────────▼──┐   ┌────▼─────┐   │
│                     │  Refiner  │   │ Packager  │   │
│                     └─────┬─────┘   └──────────┘   │
│                           │                         │
│                           └──▶ (back to Critic)     │
│                                                      │
│  Shared: ┌──────────────┐  ┌────────────────────┐   │
│          │ Groq LLM     │  │ Schema Validator   │   │
│          │ Client        │  │ (Pydantic v2)      │   │
│          └──────┬───────┘  └────────────────────┘   │
└─────────────────┼────────────────────────────────────┘
                  │ HTTPS
                  ▼
         ┌────────────────┐
         │   Groq API     │
         │ llama-3.3-70b  │
         └────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────┐
│                Output Files (Local Disk)              │
│  prompt_config.json │ architecture_spec.md │ scaff/  │
└──────────────────────────────────────────────────────┘
```

### 1.2 Component Boundaries

The LangGraph `StateGraph` is the **single orchestration primitive**. Every agent is a node function, every transition is an edge, and the Critic/Refiner loop is a conditional edge. There is no secondary orchestration layer, no custom event system, no async task queue. LangGraph handles all flow control.

**System boundary**: The engine consumes the Groq API and produces local file artifacts. It does not expose any network services, does not persist data beyond the current run (unless session history is enabled in a future version), and does not communicate with any service other than Groq.

### 1.3 Data Flow Summary

The pipeline is a transformer chain where each node enriches a shared `PipelineState` TypedDict. Data flows unidirectionally except in the Critic/Refiner cycle.

```
project_idea  ──▶  ModuleMap  ──▶  list[PromptArtifact]  ──▶  InterAgentMap
                                         │                         │
                                         ▼                         ▼
                                   list[CriticFeedback]    (fed to Critic)
                                         │
                                    ┌────▼────┐
                                    │ Pass?   │──yes──▶  FinalOutputArtifact
                                    └────┬────┘                  │
                                         │ no                    ▼
                                    Revised PromptArtifacts  Output Files
                                    (back to Critic)
```

**Immutability rule**: Agents never mutate prior agents' output in the state. The Refiner writes a new `prompt_artifacts` list (with revised versions swapped in for failed prompts). At any checkpoint, the state tells you exactly what each agent produced.

### 1.4 Why a Flat Graph (Not Nested Subgraphs)

For v1, the pipeline is simple enough that a flat graph with 6 nodes keeps the code readable. Nested subgraphs would be appropriate if individual agents had their own internal multi-step logic, but in v1 each agent is a single LLM call (or a loop over modules doing single calls). The Critic/Refiner loop is handled via a conditional edge on the flat graph, not a subgraph.

---

## 2. LangGraph State Graph Design

### 2.1 PipelineState

This is the single most important data structure in the system. It is a `TypedDict` (required by LangGraph for channel-based state management). Every field within it holds Pydantic-validated data.

```python
from typing import TypedDict, Optional

class PipelineState(TypedDict):
    # --- Input ---
    project_idea: str
    config: dict  # Serialized Settings (no API key)

    # --- Analyzer output ---
    module_map: Optional[ModuleMap]

    # --- Architect output (one per AI module) ---
    prompt_artifacts: list[PromptArtifact]

    # --- Communication Designer output ---
    inter_agent_map: Optional[InterAgentMap]

    # --- Critic output ---
    # Outer list = iterations, inner list = per-module feedback
    critic_feedback: list[list[CriticFeedback]]

    # --- Refiner tracking ---
    current_iteration: int
    max_iterations: int
    quality_threshold: float
    all_passed: bool

    # --- Best versions (for when max iterations exhausted) ---
    # module_name -> (best PromptArtifact, best score)
    best_prompt_versions: dict[str, tuple[PromptArtifact, float]]

    # --- Final output ---
    final_output: Optional[FinalOutputArtifact]

    # --- Telemetry ---
    token_usage: dict[str, int]        # agent_name -> tokens_consumed
    agent_durations: dict[str, float]  # agent_name -> seconds

    # --- Pipeline control ---
    needs_clarification: bool
    clarification_questions: list[str]

    # --- Error recovery ---
    last_checkpoint: str  # Name of last successfully completed node
    run_id: str
```

**Why `list[list[CriticFeedback]]` for iteration tracking**: The SRS requires preserving the full iteration history (FR-006.4). A nested list where `critic_feedback[0]` is iteration 1's feedback across all modules, `critic_feedback[1]` is iteration 2's, etc., provides clean indexing. The conditional edge checks `critic_feedback[-1]` to see the latest round.

**Why `best_prompt_versions` as a dict**: FR-006.5 requires using the highest-scoring version of each prompt when max iterations are exhausted. Tracking the best version per module as a `(artifact, score)` tuple avoids re-scanning the full history at packaging time.

### 2.2 Graph Nodes

Six node functions, each a pure function of `PipelineState`:

| Node Name              | Function                              | Reads from State                          | Writes to State                                                   |
| ---------------------- | ------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------- |
| `analyze`              | `analyze(state) -> dict`              | `project_idea`                            | `module_map`, `needs_clarification`, `clarification_questions`    |
| `architect`            | `architect(state) -> dict`            | `module_map`                              | `prompt_artifacts`                                                |
| `design_communication` | `design_communication(state) -> dict` | `module_map`, `prompt_artifacts`          | `inter_agent_map`                                                 |
| `critique`             | `critique(state) -> dict`             | `prompt_artifacts`, `inter_agent_map`     | `critic_feedback` (appends), `all_passed`, `best_prompt_versions` |
| `refine`               | `refine(state) -> dict`               | `prompt_artifacts`, `critic_feedback[-1]` | `prompt_artifacts` (replaces failed), `current_iteration`         |
| `package`              | `package(state) -> dict`              | All fields                                | `final_output`                                                    |

**Node functions return partial dicts, not full state.** LangGraph merges the returned dict into existing state. The `architect` node returns `{"prompt_artifacts": [...]}` and touches nothing else.

### 2.3 Graph Edges and Conditional Routing

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(PipelineState)

# --- Add nodes ---
graph.add_node("analyze", analyze)
graph.add_node("architect", architect)
graph.add_node("design_communication", design_communication)
graph.add_node("critique", critique)
graph.add_node("refine", refine)
graph.add_node("package", package)

# --- Conditional edge: Analyzer may need clarification ---
graph.add_conditional_edges(
    "analyze",
    check_clarification_needed,
    {
        "needs_clarification": END,   # Pipeline halts; CLI surfaces questions
        "proceed": "architect",
    }
)

# --- Linear edges ---
graph.add_edge("architect", "design_communication")
graph.add_edge("design_communication", "critique")
graph.add_edge("refine", "critique")  # After refining, always re-critique

# --- Conditional edge: Critic decides refine or package ---
graph.add_conditional_edges(
    "critique",
    should_continue_refining,
    {
        "refine": "refine",
        "package": "package",
    }
)

# --- Entry and exit ---
graph.set_entry_point("analyze")
graph.add_edge("package", END)
```

### 2.4 Routing Functions

```python
def check_clarification_needed(state: PipelineState) -> str:
    """Route after Analyzer: halt if input is too vague."""
    if state["needs_clarification"]:
        return "needs_clarification"
    return "proceed"


def should_continue_refining(state: PipelineState) -> str:
    """Route after Critic: refine failing prompts or proceed to packaging."""
    if state["all_passed"]:
        return "package"
    if state["current_iteration"] >= state["max_iterations"]:
        return "package"  # Use best versions; attach quality warnings
    return "refine"
```

### 2.5 Clarification Flow

When `needs_clarification` is true, the graph returns to the CLI layer with the state. The CLI displays `clarification_questions` to the user. In `prompter interactive`, the user answers and the pipeline is re-invoked with the enriched input. In `prompter generate`, the questions are printed and the process exits with code 2.

### 2.6 Visual Graph Summary

```
                    ┌─────────────┐
       START ──────▶│   analyze   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ clarify?    │
                    └──┬───────┬──┘
                  yes  │       │ no
                 ┌─────▼──┐  ┌─▼──────────┐
                 │  END   │  │  architect  │
                 └────────┘  └─────┬──────┘
                                   │
                          ┌────────▼─────────┐
                          │ design_comm      │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
              ┌────▶│          critique           │
              │     └──────────────┬──────────────┘
              │                    │
              │             ┌──────▼──────┐
              │             │ continue?   │
              │             └──┬───────┬──┘
              │            yes │       │ no
              │     ┌─────────▼──┐  ┌─▼──────────┐
              └─────│   refine   │  │  package   │
                    └────────────┘  └─────┬──────┘
                                          │
                                          ▼
                                         END
```

---

## 3. Agent Design

### 3.1 Analyzer Agent

| Property                   | Value                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Responsibility**         | Decompose the project idea into a `ModuleMap`. No template matching. First-principles reasoning.                                                             |
| **System Prompt Strategy** | Role Prompting + Output Constraints. Embeds Chain-of-Thought within instructions ("reason step-by-step about the project before producing the module list"). |
| **Temperature**            | 0.7 (creative — needs to infer modules that may not be obvious)                                                                                              |
| **Max Output Tokens**      | 4,096 (a 15-module ModuleMap serializes to ~2,500-3,500 tokens)                                                                                              |
| **Input**                  | `state["project_idea"]` (raw string)                                                                                                                         |
| **Output**                 | `ModuleMap` (Pydantic-validated)                                                                                                                             |

**System prompt key directives**:

- "Do NOT match against known project types. Decompose this specific idea from first principles." (FR-002.8)
- "If the idea is too vague to decompose into at least 3 meaningful modules, set `needs_clarification: true` and list specific questions." (FR-002.10)
- "Identify implicit requirements even if not explicitly stated." (FR-002.9)
- Output must conform to the `ModuleMap` JSON schema.

**Self-healing**: If the LLM response fails Pydantic validation, retry with the validation error appended: "Your previous response had the following schema errors: {errors}. Here is the required schema: {schema}. Fix these and respond again."

### 3.2 Architect Agent

| Property                   | Value                                                                                                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Responsibility**         | For each AI-requiring module in the `ModuleMap`, generate a `PromptArtifact`.                                                                            |
| **System Prompt Strategy** | Role Prompting ("You are a senior prompt engineer") + Few-Shot (one complete `PromptArtifact` example in the system prompt to anchor the output format). |
| **Temperature**            | 0.7 (creative)                                                                                                                                           |
| **Max Output Tokens**      | 8,192 per module                                                                                                                                         |
| **Input**                  | Full `ModuleMap` + the specific `Module` being processed                                                                                                 |
| **Output**                 | One `PromptArtifact` per LLM call                                                                                                                        |

**Sequential per-module calls**: The SRS allows parallel Architect calls (C-005), but v1 uses sequential iteration over AI modules. One LLM call per module. This stays within token limits per call and simplifies error handling. The Architect receives the full `ModuleMap` as context (so it knows about other modules) but generates one artifact at a time.

**Technique registry**: Rather than hardcoding technique names in the system prompt, techniques are defined in a `techniques.py` registry:

```python
TECHNIQUE_REGISTRY = {
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
```

The Architect's system prompt is assembled dynamically by injecting the registry contents. This satisfies NFR-004.2 (adding a technique without modifying agent code).

### 3.3 Communication Designer Agent

| Property                   | Value                                                                                                                                                                                                                   |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Responsibility**         | Produce an `InterAgentMap` from the `ModuleMap` and `prompt_artifacts`.                                                                                                                                                 |
| **System Prompt Strategy** | Role Prompting ("You are a distributed systems architect specializing in multi-agent AI systems") + Output Constraints. Structured reasoning: "For each pair of modules that share data, define the handoff condition." |
| **Temperature**            | 0.7 (creative — needs to design novel communication patterns)                                                                                                                                                           |
| **Max Output Tokens**      | 8,192                                                                                                                                                                                                                   |
| **Input**                  | `ModuleMap` + `list[PromptArtifact]`                                                                                                                                                                                    |
| **Output**                 | `InterAgentMap` (Pydantic-validated)                                                                                                                                                                                    |

**Key design point**: This agent receives the full `prompt_artifacts` list so it can reference specific context slots and output fields when defining handoffs. The system prompt instructs it to ensure every `data_inputs`/`data_outputs` from the module map is accounted for (FR-004.6).

**Minimum output requirements** (from SRS FR-004):

- Shared memory schema with all cross-agent data structures
- Handoff conditions for every agent-to-agent transition
- At least 2 context pollution prevention rules
- Trigger map covering both normal flow and error cases

### 3.4 Critic Agent

| Property                   | Value                                                                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Responsibility**         | Evaluate each `PromptArtifact` and produce a `CriticFeedback`.                                                                                                    |
| **System Prompt Strategy** | Role Prompting ("You are a prompt engineering quality assessor and red-team specialist") + Output Constraints. The system prompt embeds the exact scoring rubric. |
| **Temperature**            | 0.3 (analytical — consistent, reproducible scoring)                                                                                                               |
| **Max Output Tokens**      | 4,096 per module                                                                                                                                                  |
| **Input**                  | One `PromptArtifact` at a time, plus `ModuleMap` for context and `InterAgentMap` for communication context                                                        |
| **Output**                 | One `CriticFeedback` per call                                                                                                                                     |

**Scoring rubric** (embedded in system prompt, per FR-005.3):

| Dimension           | Weight | What it measures                                     |
| ------------------- | ------ | ---------------------------------------------------- |
| Ambiguity           | 25%    | Vague instructions the LLM could misinterpret        |
| Hallucination Risk  | 25%    | Prompts that invite the LLM to fabricate information |
| Missing Constraints | 20%    | Failure to bound the output                          |
| Edge Cases          | 15%    | Inputs or scenarios the prompt doesn't handle        |
| Token Efficiency    | 15%    | Unnecessary verbosity                                |

**Overall score** = weighted average. Prompts below the configurable threshold (default: 7.0) are sent to the Refiner.

**Why temperature 0.3**: The Critic must produce reproducible scores. A higher temperature would cause score variance across identical prompts, making the feedback loop unreliable. 0.3 balances consistency with the ability to identify subtle issues.

**Per-prompt evaluation**: The Critic evaluates one prompt at a time (not all prompts in one call). This keeps the context window manageable and allows fine-grained retry if one evaluation fails.

### 3.5 Refiner Agent

| Property                   | Value                                                                                                                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Responsibility**         | Take a `PromptArtifact` that failed the Critic and its `CriticFeedback`, produce a revised `PromptArtifact`.                                                                                               |
| **System Prompt Strategy** | Same base as Architect (Role Prompting + Few-Shot), plus: "You are revising an existing prompt based on specific Critic feedback. Address each issue listed. Do not change aspects that were not flagged." |
| **Temperature**            | 0.7 (creative rewriting)                                                                                                                                                                                   |
| **Max Output Tokens**      | 8,192                                                                                                                                                                                                      |
| **Input**                  | Original `PromptArtifact` + `CriticFeedback` for that module                                                                                                                                               |
| **Output**                 | Revised `PromptArtifact`                                                                                                                                                                                   |

**Key requirement (FR-006.6)**: The Refiner receives both the original prompt AND the feedback — not just the feedback alone. The LLM input format is:

```
System: [Refiner system prompt]
User: Original prompt artifact: {artifact JSON}

Critic feedback: {feedback JSON}

Revise the prompt to address all flagged issues while maintaining
the prompt's core purpose, technique selection, and context slots.
Return the complete revised PromptArtifact.
```

**Best-version tracking**: After each Refiner cycle, the Critic re-evaluates. If the new score is higher than the previous best for that module, `best_prompt_versions[module_name]` is updated. When max iterations are exhausted without all prompts passing, the Packager uses `best_prompt_versions` (the highest-scoring version of each prompt, not the last version).

### 3.6 Packager Agent

| Property                   | Value                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| **Responsibility**         | Assemble `FinalOutputArtifact`, generate the Markdown spec, and produce code scaffolding.           |
| **System Prompt Strategy** | Output Constraints. The Packager must not modify prompt text (FR-007.7) — only format and assemble. |
| **Temperature**            | 0.5 (balanced)                                                                                      |
| **Max Output Tokens**      | 16,384 (the Markdown spec can be long)                                                              |
| **Input**                  | All state fields                                                                                    |
| **Output**                 | `FinalOutputArtifact` + side-effect file writes to disk                                             |

**Split into LLM call + deterministic assembly**:

| Output Artifact                              | Generated By | Method                                                                          |
| -------------------------------------------- | ------------ | ------------------------------------------------------------------------------- |
| `prompt_config.json`                         | Python code  | Deterministic Pydantic `.model_dump_json()` — the LLM never touches JSON output |
| `architecture_spec.md` (narrative sections)  | LLM          | Executive summary, module descriptions, implementation notes                    |
| `architecture_spec.md` (structured sections) | Python code  | Tables, prompt text blocks, quality scores, token budgets                       |
| `scaffolding/`                               | Python code  | Template-based file generation using Jinja2 or string formatting                |

**Why deterministic JSON assembly**: The LLM could corrupt prompt text during packaging. Assembling `FinalOutputArtifact` from Pydantic models via Python eliminates this risk and satisfies FR-007.7.

**Multi-call splitting for large projects (SRS C-002)**: If there are more than 8 AI modules, the Markdown narrative generation is split into batches:

1. One call for the executive summary + first 4 modules
2. Subsequent calls for remaining modules (4 at a time)
3. A final call for the communication overview and quality summary

Results are concatenated deterministically.

### 3.7 Agent LLM Configuration Summary

| Agent                  | Temperature | Max Tokens | Calls per Run            | JSON Mode                   |
| ---------------------- | ----------- | ---------- | ------------------------ | --------------------------- |
| Analyzer               | 0.7         | 4,096      | 1                        | Yes                         |
| Architect              | 0.7         | 8,192      | N (one per AI module)    | Yes                         |
| Communication Designer | 0.7         | 8,192      | 1                        | Yes                         |
| Critic                 | 0.3         | 4,096      | N per iteration          | Yes                         |
| Refiner                | 0.7         | 8,192      | <=N per iteration        | Yes                         |
| Packager               | 0.5         | 16,384     | 1-3 (Markdown narrative) | No (free text for Markdown) |

---

## 4. State Management

### 4.1 State Immutability Principle

Agents never mutate prior agents' output in the state. The Refiner writes to `prompt_artifacts`, but it replaces the entire list (with revised versions of failed prompts swapped in). This makes debugging straightforward: at any checkpoint, the state is a complete snapshot.

### 4.2 State Serialization for Checkpoints

After each node completes, the pipeline state is serialized to disk:

```
.prompter_state/
  <run_id>/
    pipeline_state.json     # Full PipelineState snapshot
    agent_logs.jsonl        # Per-agent invocation logs (append-only)
```

Serialization:

- Pydantic models serialize via `.model_dump()`.
- The checkpoint file includes: `run_id`, `timestamp`, `last_completed_node`, and the full state.
- API keys are excluded from serialized state.

Resumption (`prompter generate --resume <path>`):

- Deserialize JSON, validate each field against its Pydantic model.
- Re-enter the graph at the node following `last_checkpoint`.

**Why file-based checkpointing over SQLite**: SQLite adds a dependency and complexity for a single-user, single-run tool. JSON files in a known directory are simpler, debuggable by humans, and sufficient for v1.

### 4.3 State Size Budget

Estimated state size for a 10-module project:

| Component                                     | Estimated Size |
| --------------------------------------------- | -------------- |
| `ModuleMap`                                   | ~5 KB          |
| `prompt_artifacts` (10 modules)               | ~50 KB         |
| `inter_agent_map`                             | ~10 KB         |
| `critic_feedback` (3 iterations x 10 modules) | ~30 KB         |
| Telemetry + metadata                          | ~5 KB          |
| **Total**                                     | **~100 KB**    |

Well within memory and disk constraints.

### 4.4 Telemetry Accumulation

`token_usage` and `agent_durations` are accumulated across nodes. Each node function reads the current values, adds its own contribution, and returns the updated dict. These are used for the pipeline summary at completion (NFR-006.3).

```python
# Inside each node function:
start_time = time.time()
# ... do work, track tokens ...
duration = time.time() - start_time

return {
    "some_output": result,
    "token_usage": {**state["token_usage"], "analyzer": tokens_used},
    "agent_durations": {**state["agent_durations"], "analyzer": duration},
    "last_checkpoint": "analyze",
}
```

---

## 5. Error Handling and Resilience

### 5.1 Three-Layer Retry Strategy

Three distinct retry layers, each with its own counter and strategy:

| Layer                 | What it Handles                            | Strategy                                                                 | Implementation                                          |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| **HTTP Transport**    | Network errors, timeouts, 5xx responses    | 3 retries, exponential backoff (1s, 2s, 4s) with jitter                  | `tenacity` library wrapping the Groq client call        |
| **Rate Limiting**     | HTTP 429 responses                         | Wait for `retry-after` header duration; if absent, backoff 10s, 20s, 40s | Custom middleware in the Groq client wrapper            |
| **Schema Validation** | LLM returns invalid or non-conforming JSON | Append validation error to prompt, retry up to 2 times (self-healing)    | In each node function, catch Pydantic `ValidationError` |

**Why three separate layers**: HTTP errors and schema errors require different recovery strategies. Conflating them into a single retry counter would exhaust retries on schema errors when the network is fine, or vice versa.

### 5.2 Self-Healing Schema Validation

When the LLM returns JSON that fails Pydantic validation:

1. Capture the `ValidationError` messages.
2. Construct a correction message:

    ```
    Your previous response had these schema errors:
    {validation_errors}

    Here is the required JSON schema:
    {model.model_json_schema()}

    Respond with corrected JSON that conforms to this schema.
    ```

3. Re-call the LLM with: original system prompt + original user message + error correction message.
4. If the second attempt also fails, try once more (3 total: 1 original + 2 retries).
5. After 3 total attempts, raise a `SchemaValidationError` that halts the pipeline with a clear error message.

**Including the full JSON schema in the retry prompt**: The model may have incorrectly inferred the schema from the system prompt. Providing the explicit JSON schema in the retry dramatically increases the chance of a valid response.

### 5.3 Rate Limit Pre-Flight Check

Before starting the pipeline, estimate the total number of LLM calls and tokens:

| Agent                  | Calls             | Est. Tokens per Call |
| ---------------------- | ----------------- | -------------------- |
| Analyzer               | 1                 | ~4,000               |
| Architect              | N (AI modules)    | ~8,000               |
| Communication Designer | 1                 | ~8,000               |
| Critic                 | N per iteration   | ~4,000               |
| Refiner                | <=N per iteration | ~8,000               |
| Packager               | 1-3               | ~16,000              |

**Worst case for 10 AI modules, 3 iterations**: 1 + 10 + 1 + 30 + 20 + 3 = **65 calls, ~400K tokens**. Against Groq rate limits (300K tokens/min, 1K requests/min), this may hit token-per-minute limits.

If the estimate exceeds 80% of the per-minute rate, warn the user:

```
Warning: This project may experience rate-limit delays.
Estimated: ~400K tokens across ~65 API calls.
Groq limit: 300K tokens/minute.
The pipeline will automatically pause and retry if limits are hit.
```

### 5.4 Graceful Degradation

If a node fails after all retries:

1. Save the current state checkpoint to `.prompter_state/<run_id>/`.
2. Print a clear error:

    ```
    Pipeline failed at [agent_name].
    Error: [error description]

    State saved to: .prompter_state/<run_id>/pipeline_state.json
    Resume with: prompter generate --resume .prompter_state/<run_id>/pipeline_state.json
    ```

3. Exit with code 1.

### 5.5 LLM Call Timeout

Individual LLM calls timeout after 60 seconds (NFR-001.2). The timeout is enforced by the Groq client wrapper. On timeout, the call is retried using the HTTP transport retry layer.

---

## 6. Project Source Code Structure

```
prompter/
├── __init__.py
├── __main__.py                     # Entry point: python -m prompter
├── cli.py                          # Typer CLI definition
│
├── graph.py                        # LangGraph StateGraph construction
├── state.py                        # PipelineState TypedDict + helpers
│
├── agents/                         # Agent node functions
│   ├── __init__.py
│   ├── analyzer.py                 # analyze() node function
│   ├── architect.py                # architect() node function
│   ├── communication_designer.py   # design_communication() node function
│   ├── critic.py                   # critique() node function
│   ├── refiner.py                  # refine() node function
│   └── packager.py                 # package() node function
│
├── models/                         # Pydantic v2 data models
│   ├── __init__.py                 # Re-exports all models
│   ├── module_map.py               # ModuleMap, Module, DomainClassification
│   ├── prompt_artifact.py          # PromptArtifact, ContextSlot, TokenEstimate
│   ├── inter_agent_map.py          # InterAgentMap, SharedMemoryField, etc.
│   ├── critic_feedback.py          # CriticFeedback, Issue, CategoryScore
│   └── final_output.py             # FinalOutputArtifact, PipelineMetadata
│
├── llm/                            # LLM client layer
│   ├── __init__.py
│   ├── client.py                   # Groq client wrapper (retry, rate limit, tracking)
│   ├── prompts.py                  # System prompt loader (reads from prompt_templates/)
│   └── techniques.py               # Technique registry (CoT, Few-Shot, etc.)
│
├── output/                         # Output artifact generation
│   ├── __init__.py
│   ├── json_writer.py              # Write FinalOutputArtifact to JSON
│   ├── markdown_writer.py          # Generate Markdown spec
│   └── scaffold_writer.py          # Generate Python scaffolding files
│
├── config.py                       # Settings via pydantic-settings
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── tokens.py                   # Token counting (tiktoken)
│   ├── checkpoint.py               # State serialization/deserialization
│   └── logging.py                  # Structured logging setup
│
└── prompt_templates/               # Agent system prompts (text files)
    ├── analyzer_system.txt
    ├── architect_system.txt
    ├── communication_designer_system.txt
    ├── critic_system.txt
    ├── refiner_system.txt
    └── packager_system.txt

tests/
├── __init__.py
├── conftest.py                     # Shared fixtures (mock LLM, sample data)
├── unit/
│   ├── test_models.py              # Pydantic model validation
│   ├── test_state.py               # State management
│   ├── test_token_counting.py
│   └── test_config.py
├── integration/
│   ├── test_analyzer.py            # Individual agent with mocked LLM
│   ├── test_architect.py
│   ├── test_critic.py
│   ├── test_refiner.py
│   ├── test_packager.py
│   └── test_pipeline.py            # Full pipeline with mocked LLM
└── regression/
    ├── test_prompt_quality.py       # Golden-file tests (real API)
    └── fixtures/
        ├── quizzing_platform/       # Known-good inputs and expected outputs
        └── customer_support/

pyproject.toml                       # Project metadata, deps, CLI entry point
.env.example                         # Template for environment variables
.gitignore
README.md
```

### 6.1 Key Structural Decisions

**Prompt templates as separate text files**: System prompts are long (up to 2,000 tokens per NFR-003.1). Text files with `{{variable}}` placeholders are easier to read, edit, version-control, and diff than inline Python strings.

**`models/` separate from `agents/`**: The Pydantic models are the data contracts between agents. A separate package prevents circular imports and makes contracts easy to find, review, and test independently.

**`llm/client.py` as a single wrapper**: All agents call one function: `call_llm(system_prompt, user_message, temperature, max_tokens, response_model)`. This centralizes retry logic, rate limiting, JSON mode enforcement, token tracking, and logging. No agent calls the Groq API directly.

---

## 7. Configuration Management

### 7.1 Configuration Hierarchy

Three sources, in priority order (higher overrides lower):

1. **CLI flags**: `--quality-threshold 8.0`, `--max-iterations 5`, etc.
2. **Environment variables**: `GROQ_API_KEY`, `PROMPTER_QUALITY_THRESHOLD`, etc.
3. **Defaults**: Hardcoded in the `Settings` Pydantic model.

### 7.2 Settings Model

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- API ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- Pipeline ---
    quality_threshold: float = 7.0
    max_iterations: int = 3
    per_prompt_token_threshold: int = 8000
    pipeline_token_threshold: int = 50000

    # --- Temperatures ---
    creative_temperature: float = 0.7
    analytical_temperature: float = 0.3
    packager_temperature: float = 0.5

    # --- Retry ---
    max_retries: int = 3
    retry_base_delay: float = 1.0
    schema_retry_limit: int = 2
    llm_timeout_seconds: int = 60

    # --- Output ---
    output_dir: str = "./output"
    scaffold_enabled: bool = True
    output_format: str = "both"  # json | markdown | both

    # --- Logging ---
    verbose: bool = False

    model_config = {"env_prefix": "PROMPTER_"}
```

### 7.3 Security

- `groq_api_key` is loaded only from the `GROQ_API_KEY` environment variable (NFR-005.1).
- The `Settings` model overrides `__repr__` to mask the API key: `groq_api_key='***'`.
- Checkpoint files never serialize the API key (excluded from `PipelineState`).
- `.env` is listed in `.gitignore` (NFR-005.4).
- Generated outputs never contain the engine's own internal system prompts (NFR-005.2).

### 7.4 Why No Config File for v1

CLI flags + environment variables are sufficient for the v1 parameter set. A config file (YAML/TOML) adds complexity: file discovery, conflict resolution with CLI flags, and schema validation. If the parameter set grows significantly in v2, introduce a `~/.prompter/config.toml`.

---

## 8. Testing Strategy

### 8.1 Test Pyramid

| Layer                            | Count     | Speed     | What It Tests                                                                                      |
| -------------------------------- | --------- | --------- | -------------------------------------------------------------------------------------------------- |
| **Unit**                         | ~40 tests | <1s each  | Pydantic model validation, token counting, config parsing, state serialization, technique registry |
| **Integration** (mocked LLM)     | ~15 tests | <2s each  | Each agent node with deterministic LLM responses, full pipeline with mocked calls, error recovery  |
| **Prompt Regression** (real API) | ~5 tests  | ~60s each | Golden-file comparison for known inputs; detects model behavior drift                              |

### 8.2 LLM Mocking Strategy

The `llm/client.py` wrapper accepts a `client` parameter. In tests, inject a mock client that returns pre-recorded JSON responses:

```
tests/integration/fixtures/
    analyzer_response_quizzing.json
    architect_response_question_gen.json
    critic_response_high_score.json
    critic_response_low_score.json
    ...
```

**Record-and-replay**: Generate fixtures by running the pipeline once with a real API key and saving the responses. This ensures fixtures are realistic. When the model changes, re-record and update golden files.

### 8.3 Prompt Regression Tests

The highest-value tests and the most architecturally significant:

1. Run the full pipeline against the real Groq API with a fixed input (e.g., "a quizzing platform for medical students with adaptive difficulty").
2. Compare the output **structure** (not exact text) against golden files:
    - Module count is within expected range (4-8).
    - Required modules are present (question generation, adaptive difficulty).
    - All prompts have quality scores above 6.0.
    - InterAgentMap has at least 3 shared memory fields.
3. These tests detect silent model behavior changes (SRS A-004).
4. Run in CI on a schedule (weekly), not on every commit.

### 8.4 Schema Validation Tests

For each Pydantic model, test:

- Valid data passes validation.
- Missing required fields raise `ValidationError`.
- Wrong types raise `ValidationError`.
- Edge cases: empty module list, zero AI modules, maximum 15 modules, scores at boundary values (0, 7.0, 10).

### 8.5 Error Scenario Tests

- Groq API returns 429 (rate limit) — verify exponential backoff.
- Groq API returns 500 — verify 3 retries then fail.
- LLM returns malformed JSON — verify self-healing retry.
- LLM returns valid JSON that fails Pydantic validation — verify schema error appended and retried.
- Critic scores all below threshold for 3 iterations — verify best versions used and warnings attached.
- Pipeline interrupted mid-execution — verify checkpoint saved and `--resume` works.

---

## 9. Deployment and Packaging

### 9.1 Python Package

```toml
# pyproject.toml
[project]
name = "prompter"
version = "1.0.0"
description = "A multi-agent AI system that builds prompt architectures from project ideas"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain-groq>=0.2.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "typer>=0.9.0",
    "rich>=13.0",
    "tiktoken>=0.5.0",
    "tenacity>=8.0",
    "python-dotenv>=1.0",
]

[project.scripts]
prompter = "prompter.cli:app"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
```

### 9.2 Why Typer Over Click

Typer is built on Click but provides type-hint-based argument parsing, aligning with the Pydantic-first philosophy of the project. It reduces boilerplate for CLI argument definitions.

### 9.3 Why Rich for CLI Output

The SRS requires progress indicators (NFR-001.3) and formatted output. `rich` provides spinners, progress bars, tables, and syntax-highlighted JSON output.

### 9.4 Installation

```bash
# Clone and install
git clone <repo>
cd prompter
pip install -e ".[dev]"

# Set API key
export GROQ_API_KEY="your-key-here"
# Or: create .env file with GROQ_API_KEY=your-key-here

# Run
prompter generate "a quizzing platform for medical students"
```

---

## 10. Key Architectural Decisions

These decisions separate this system from "call the LLM five times in a loop and concatenate the results."

### Decision 1: Three-Layer Error Recovery

A naive implementation retries on network errors and gives up on bad JSON. This architecture treats schema validation failures as recoverable by feeding the error back to the LLM with the explicit JSON schema. A second attempt with error context almost always succeeds because the LLM can self-correct when shown its mistakes.

### Decision 2: State Checkpoint and Resume

A naive implementation loses all progress when a call fails at step 4 of 5. This architecture writes a checkpoint after every node, enabling `--resume`. For a pipeline that takes 2-5 minutes and costs real API tokens, this prevents users from paying twice for the same computation.

### Decision 3: Best-Version Tracking in the Critic/Refiner Loop

A naive implementation either takes the last version (which may have regressed on some dimensions while improving others) or the first passing version. This architecture tracks the best score per module across all iterations and selects the best version, not the last version, when iterations are exhausted. This prevents the case where iteration 2 scores 6.8 and iteration 3 scores 6.2 but iteration 2 is discarded.

### Decision 4: LLM-Free Packaging

A naive implementation asks the LLM to produce the final JSON and Markdown. This architecture uses the LLM only for narrative generation (executive summary, implementation notes) and performs all structured assembly in deterministic Python. This eliminates the risk of LLM-corrupted prompt text during packaging (FR-007.7) and removes a failure mode where the Packager's output exceeds token limits.

### Decision 5: Technique Registry for Extensibility

A naive implementation hardcodes technique names in the Architect's system prompt. This architecture externalizes techniques into a registry (`techniques.py`), assembles the Architect's prompt dynamically, and satisfies NFR-004.2 (adding a technique without modifying agent code).

### Decision 6: Separate Temperatures Per Agent Role

A naive implementation uses a single temperature for all calls. This architecture uses 0.7 for creative agents, 0.3 for the Critic, and 0.5 for the Packager. The Critic's low temperature ensures scoring consistency, which is essential for the feedback loop to converge rather than oscillate.

### Decision 7: Per-Module LLM Calls Over Batch Calls

A naive implementation tries to generate all prompts in one call. This architecture generates one prompt per LLM call. This keeps each call within token limits, allows fine-grained retry (one module's failure does not lose others), and produces cleaner outputs because the model can focus on one module at a time.

### Decision 8: Pydantic Boundary Enforcement at Every Stage

A naive implementation parses JSON and hopes for the best. This architecture validates every LLM response against a Pydantic model before passing it to the next node. The contracts are defined in `models/`, testable independently, and serve as the source of truth for both runtime validation and LLM output guidance (JSON schema injected into retry prompts).

---

_End of System Architecture Document._
