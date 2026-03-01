# Development Orchestration Plan

**Purpose**: A phased execution plan for building the Prompt Engineering Engine autonomously, from zero code to a working CLI tool.

**References**: [ARCHITECTURE.md](./ARCHITECTURE.md) | [SRS.md](./SRS.md) | [PRD.md](./PRD.md)

---

## Phasing Overview

```
Phase 0: Project Scaffolding           ──  project structure, deps, git
Phase 1: Foundation Layer              ──  models, config, LLM client, utils
Phase 2: Analyzer Agent                ──  first agent + CLI intake
Phase 3: Architect Agent               ──  technique registry, per-module gen
Phase 4: Communication Designer        ──  inter-agent map generation
Phase 5: Critic + Refiner Loop         ──  quality loop with conditional edges
Phase 6: Packager + Output             ──  JSON, Markdown, scaffolding writers
Phase 7: LangGraph Integration         ──  wire everything, full pipeline
Phase 8: Testing + Polish              ──  tests, error scenarios, CLI polish
```

Each phase is buildable and verifiable independently. Phases 2-6 can be partially developed in isolation because they share the same foundation (Phase 1) and only come together in Phase 7.

---

## Phase 0: Project Scaffolding

### Goal

Create the project skeleton so that `pip install -e .` works and `prompter --help` prints usage.

### Files to Create

| File                   | Purpose                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `pyproject.toml`       | Package metadata, dependencies, `[project.scripts]` entry point                    |
| `.gitignore`           | Python, `.env`, `.prompter_state/`, IDE files                                      |
| `.env.example`         | Template: `GROQ_API_KEY=your-key-here`                                             |
| `prompter/__init__.py` | Package init with `__version__`                                                    |
| `prompter/__main__.py` | `from prompter.cli import app; app()`                                              |
| `prompter/cli.py`      | Typer app with `generate` and `interactive` stubs that print "not yet implemented" |

### Dependencies to Install

```
langgraph, langchain-groq, pydantic, pydantic-settings,
typer, rich, tiktoken, tenacity, python-dotenv
```

Dev: `pytest, pytest-cov, ruff, mypy`

### Verification

- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `prompter --help` prints usage
- [ ] `prompter --version` prints version
- [ ] `prompter generate "test"` prints "not yet implemented"
- [ ] `git init && git add . && git commit` succeeds
- [ ] `.env` is NOT tracked by git

### Dependencies

None (starting point).

---

## Phase 1: Foundation Layer

### Goal

Build all shared infrastructure that agents depend on: data models, configuration, LLM client, token counting, and state management.

### Files to Create

| File                                 | Purpose                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `prompter/models/__init__.py`        | Re-export all models                                                                        |
| `prompter/models/module_map.py`      | `ModuleMap`, `Module`, `DomainClassification`, `InteractionType`                            |
| `prompter/models/prompt_artifact.py` | `PromptArtifact`, `ContextSlot`, `TokenEstimate`, `PromptChain`, `EvalCriteria`             |
| `prompter/models/inter_agent_map.py` | `InterAgentMap`, `SharedMemoryField`, `HandoffCondition`, `ContextPollutionRule`, `Trigger` |
| `prompter/models/critic_feedback.py` | `CriticFeedback`, `Issue`, `CategoryScore`, `Severity`, `IssueCategory`                     |
| `prompter/models/final_output.py`    | `FinalOutputArtifact`, `PipelineMetadata`                                                   |
| `prompter/state.py`                  | `PipelineState` TypedDict, `create_initial_state()` helper                                  |
| `prompter/config.py`                 | `Settings` model via `pydantic-settings`                                                    |
| `prompter/llm/__init__.py`           | Package init                                                                                |
| `prompter/llm/client.py`             | `call_llm()` wrapper with three-layer retry, JSON mode, token tracking                      |
| `prompter/llm/techniques.py`         | `TECHNIQUE_REGISTRY` dict                                                                   |
| `prompter/llm/prompts.py`            | `load_prompt(template_name)` — reads from `prompt_templates/`                               |
| `prompter/utils/__init__.py`         | Package init                                                                                |
| `prompter/utils/tokens.py`           | `estimate_tokens(text)` using tiktoken                                                      |
| `prompter/utils/checkpoint.py`       | `save_checkpoint(state, path)`, `load_checkpoint(path)`                                     |
| `prompter/utils/logging.py`          | `setup_logging(verbose)` with rich handler                                                  |
| `tests/conftest.py`                  | Shared fixtures: mock LLM client, sample ModuleMap, sample PromptArtifact                   |
| `tests/unit/test_models.py`          | Validation tests for all Pydantic models                                                    |
| `tests/unit/test_config.py`          | Settings loading from env vars                                                              |
| `tests/unit/test_token_counting.py`  | tiktoken estimation accuracy                                                                |

### Key Implementation Details

**`call_llm()` signature**:

```python
def call_llm(
    system_prompt: str,
    user_message: str,
    response_model: type[BaseModel],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    client: Optional[ChatGroq] = None,  # Injectable for testing
) -> BaseModel:
```

This function:

1. Calls the Groq API with `response_format={"type": "json_object"}`
2. Parses response as JSON
3. Validates against `response_model` (Pydantic)
4. On `ValidationError`: appends error + schema to messages, retries (up to `schema_retry_limit`)
5. On HTTP error: retries with exponential backoff (up to `max_retries`)
6. On 429: waits for `retry-after` or uses increasing backoff
7. Tracks tokens consumed and latency
8. Returns the validated Pydantic model instance

**Technique registry**: A dict of technique definitions. The Architect pulls from this at runtime to assemble its system prompt.

### Verification

- [ ] All Pydantic models accept valid data and reject invalid data (unit tests pass)
- [ ] `Settings` loads `GROQ_API_KEY` from environment
- [ ] `call_llm()` with a mock client returns parsed Pydantic models
- [ ] `call_llm()` with an intentionally bad response triggers self-healing retry
- [ ] `estimate_tokens("hello world")` returns a reasonable number
- [ ] `save_checkpoint` / `load_checkpoint` round-trips a `PipelineState`
- [ ] `pytest tests/unit/` passes

### Dependencies

Phase 0 completed.

---

## Phase 2: Analyzer Agent

### Goal

The first working agent. Given a project idea, produce a `ModuleMap`. Wire into the CLI so `prompter generate "idea"` runs just the Analyzer and prints the result.

### Files to Create

| File                                            | Purpose                                               |
| ----------------------------------------------- | ----------------------------------------------------- |
| `prompter/prompt_templates/analyzer_system.txt` | Analyzer system prompt (full text)                    |
| `prompter/agents/__init__.py`                   | Package init                                          |
| `prompter/agents/analyzer.py`                   | `analyze(state: PipelineState) -> dict` node function |

### Files to Modify

| File              | Change                                                               |
| ----------------- | -------------------------------------------------------------------- |
| `prompter/cli.py` | Wire `generate` command to call the Analyzer and print the ModuleMap |

### System Prompt Design (analyzer_system.txt)

The Analyzer prompt must:

- Define the role: "You are an expert software architect and AI systems designer."
- Forbid template matching: "Do NOT match against known project types. Decompose from first principles."
- Require step-by-step reasoning before producing the module list
- Instruct: identify implicit requirements (FR-002.9)
- Instruct: set `needs_clarification` if the idea is too vague (FR-002.10)
- Specify the exact JSON output schema (ModuleMap)
- Constrain module count: 3-15

### Verification

- [ ] `prompter generate "a quizzing platform for medical students"` produces a ModuleMap with 4-8 modules
- [ ] The output includes domain classification (Education/Healthcare)
- [ ] Each module has: name, description, `requires_ai`, `data_inputs`, `data_outputs`
- [ ] Given a vague input like "an app", the Analyzer sets `needs_clarification: true` with questions
- [ ] The ModuleMap passes Pydantic validation
- [ ] Output is displayed cleanly in the terminal via `rich`

### Dependencies

Phase 1 completed.

---

## Phase 3: Architect Agent

### Goal

For each AI-requiring module in the ModuleMap, generate a `PromptArtifact` with technique selection, context slots, and eval criteria.

### Files to Create

| File                                             | Purpose                                                   |
| ------------------------------------------------ | --------------------------------------------------------- |
| `prompter/prompt_templates/architect_system.txt` | Architect system prompt (with technique catalog injected) |
| `prompter/agents/architect.py`                   | `architect(state: PipelineState) -> dict` node function   |

### Key Implementation Details

**Per-module iteration**: The `architect()` function loops over `module_map.modules` where `requires_ai == True`, calls `call_llm()` once per module, and collects results into `list[PromptArtifact]`.

**Context injection**: Each call receives:

- System prompt: Architect system prompt + technique registry (dynamically assembled)
- User message: Full `ModuleMap` JSON + the specific `Module` JSON being processed

**Technique selection**: The system prompt tells the LLM to choose from the technique registry and provide a rationale. The LLM fills `primary_technique` and optionally `secondary_technique` fields.

### Verification

- [ ] Given a 5-module ModuleMap (3 AI modules), produces 3 `PromptArtifact`s
- [ ] Each artifact has: system prompt text, technique selection, rationale, context slots, token estimate, eval criteria
- [ ] Technique selections are from the valid set (CoT, Few-Shot, Role Prompting, Output Constraints, Socratic)
- [ ] Context slots use `{{variable}}` syntax
- [ ] Each artifact has at least 2 good/bad output examples and 3 eval suggestions
- [ ] Integration test with mocked LLM passes

### Dependencies

Phase 2 completed (needs a working ModuleMap from the Analyzer).

---

## Phase 4: Communication Designer Agent

### Goal

Generate the `InterAgentMap` defining how modules communicate: shared memory, handoffs, triggers, and pollution prevention.

### Files to Create

| File                                                          | Purpose                                              |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| `prompter/prompt_templates/communication_designer_system.txt` | System prompt                                        |
| `prompter/agents/communication_designer.py`                   | `design_communication(state: PipelineState) -> dict` |

### Key Implementation Details

The Communication Designer receives both `ModuleMap` and `list[PromptArtifact]` so it can reference specific context slots and outputs when defining handoffs.

**Validation rule (FR-004.6)**: After the LLM produces the `InterAgentMap`, a post-processing step checks that every `data_inputs`/`data_outputs` from the module map appears in either a shared memory field or a handoff condition. Missing references are flagged as warnings.

### Verification

- [ ] Produces a shared memory schema with at least 3 fields for a 5-module project
- [ ] Handoff conditions exist for every module-to-module transition
- [ ] At least 2 context pollution prevention rules
- [ ] Trigger map covers normal flow and error cases
- [ ] Every `data_inputs`/`data_outputs` is accounted for (validation check passes)
- [ ] Integration test with mocked LLM passes

### Dependencies

Phase 3 completed (needs PromptArtifacts).

---

## Phase 5: Critic + Refiner Loop

### Goal

Implement the quality assurance feedback loop: the Critic scores prompts, the Refiner fixes failing ones, and the loop iterates until thresholds are met or max iterations are reached.

### Files to Create

| File                                           | Purpose                                              |
| ---------------------------------------------- | ---------------------------------------------------- |
| `prompter/prompt_templates/critic_system.txt`  | Critic system prompt with scoring rubric and weights |
| `prompter/prompt_templates/refiner_system.txt` | Refiner system prompt                                |
| `prompter/agents/critic.py`                    | `critique(state: PipelineState) -> dict`             |
| `prompter/agents/refiner.py`                   | `refine(state: PipelineState) -> dict`               |

### Key Implementation Details

**Critic node**:

- Iterates over `prompt_artifacts`, calls `call_llm()` once per prompt
- Produces `list[CriticFeedback]` (one per module)
- Computes `all_passed` (all scores >= threshold)
- Updates `best_prompt_versions` if current score beats previous best

**Refiner node**:

- Reads `critic_feedback[-1]` (latest round)
- For each prompt that failed: calls `call_llm()` with original artifact + feedback
- Produces revised `PromptArtifact`
- Replaces failed prompts in `prompt_artifacts` list (passing prompts are kept as-is)
- Increments `current_iteration`

**Best-version tracking**: Maintained in `best_prompt_versions`. When max iterations are exhausted, the Packager reads from this dict instead of `prompt_artifacts`.

### Verification

- [ ] Critic scores a deliberately weak prompt below 7.0
- [ ] Critic scores a strong prompt above 7.0
- [ ] Scores include all 5 dimensions with correct weights
- [ ] Issues have: category, severity, location (quoted text), description, suggestion
- [ ] Refiner produces a revised prompt that addresses flagged issues
- [ ] Revised prompt scores higher than original on re-evaluation
- [ ] Loop terminates when all pass threshold
- [ ] Loop terminates at max iterations and uses best versions
- [ ] Iteration history is preserved in `critic_feedback`
- [ ] Integration test with mocked LLM (low-score then high-score responses) passes

### Dependencies

Phase 3 completed (needs PromptArtifacts to critique). Phase 4 is independent — Critic can run with or without InterAgentMap.

---

## Phase 6: Packager + Output

### Goal

Assemble all pipeline outputs into the three deliverable artifacts: JSON config, Markdown spec, and Python scaffolding.

### Files to Create

| File                                            | Purpose                                                 |
| ----------------------------------------------- | ------------------------------------------------------- |
| `prompter/prompt_templates/packager_system.txt` | System prompt for Markdown narrative generation         |
| `prompter/agents/packager.py`                   | `package(state: PipelineState) -> dict`                 |
| `prompter/output/__init__.py`                   | Package init                                            |
| `prompter/output/json_writer.py`                | `write_json(artifact: FinalOutputArtifact, path: Path)` |
| `prompter/output/markdown_writer.py`            | `write_markdown(state: PipelineState, path: Path)`      |
| `prompter/output/scaffold_writer.py`            | `write_scaffolding(state: PipelineState, path: Path)`   |

### Key Implementation Details

**JSON writer**: Pure Python. `FinalOutputArtifact.model_dump_json(indent=2)` written to `prompt_config.json`. The LLM never touches this output.

**Markdown writer**: Hybrid approach.

- LLM generates: executive summary, per-module narrative descriptions, implementation notes
- Python generates: table of contents, prompt text code blocks, quality score tables, token budget tables, communication design tables
- Sections are interleaved and written to `architecture_spec.md`

**Scaffold writer**: Deterministic Python template generation.

- `scaffolding/README.md` — setup instructions
- `scaffolding/prompts/<module_name>.txt` — one prompt file per module (extracted from PromptArtifact)
- `scaffolding/agents/<module_name>_agent.py` — agent stub with the system prompt loaded
- `scaffolding/config.py` — configuration template
- `scaffolding/main.py` — entry point template with orchestration wiring

**Multi-call splitting**: If >8 AI modules, Markdown narrative is generated in batches.

### Verification

- [ ] `prompt_config.json` passes schema validation (`FinalOutputArtifact.model_validate_json()`)
- [ ] `architecture_spec.md` renders correctly in a Markdown viewer
- [ ] Markdown includes: executive summary, module breakdown, communication design, quality scores
- [ ] `scaffolding/` contains: README, prompts/, agents/, config.py, main.py
- [ ] Python scaffolding files have no syntax errors (`py_compile`)
- [ ] Prompts below 7.0 are flagged with warnings in both JSON and Markdown
- [ ] Output directory structure matches SRS Section 6.3

### Dependencies

Phases 1-5 completed (needs all state fields populated).

---

## Phase 7: LangGraph Integration

### Goal

Wire all agents into the LangGraph StateGraph with edges, conditional routing, checkpoint/resume, and the full CLI experience.

### Files to Create

| File                | Purpose                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `prompter/graph.py` | `build_graph() -> CompiledGraph` — complete StateGraph with all nodes and edges |

### Files to Modify

| File                           | Change                                                       |
| ------------------------------ | ------------------------------------------------------------ |
| `prompter/cli.py`              | Full implementation of `generate` and `interactive` commands |
| `prompter/utils/checkpoint.py` | Wire checkpointing into the graph execution                  |

### Key Implementation Details

**`build_graph()`**:

- Adds all 6 nodes
- Adds conditional edge after `analyze` (clarification check)
- Adds linear edges: `architect` -> `design_communication` -> `critique`
- Adds conditional edge after `critique` (refine or package)
- Adds edge: `refine` -> `critique` (loop back)
- Sets entry point and END edges

**CLI `generate` command**:

1. Parse input (inline text or file)
2. Validate input (10-10,000 chars, non-empty)
3. Load `Settings`
4. Create initial `PipelineState`
5. Build and invoke graph
6. If `needs_clarification`, print questions and exit (code 2)
7. Display progress with `rich` (spinner per agent, score table for Critic)
8. Print output summary
9. Save checkpoint after each node (for `--resume`)

**CLI `interactive` command**:

- Same as `generate` but with a confirmation gate after the Analyzer
- Display module list, prompt: `Proceed? [Y/n/edit]`
- On `edit`: allow adding/removing/modifying modules inline
- After confirmation, run the rest of the pipeline

**Progress display**:

```
[spinner] Analyzing project idea...                    done (3.2s)
[spinner] Generating prompts (module 1/5)...           done (4.1s)
[spinner] Generating prompts (module 2/5)...           done (3.8s)
...
[spinner] Designing inter-agent communication...       done (5.1s)
[table]
  Module: Question Generator     Score: 8.2/10  PASS
  Module: Difficulty Adapter      Score: 6.5/10  REFINING...
...
[spinner] Refining (iteration 1)...                    done (4.5s)
[table] Updated scores...
[spinner] Packaging output...                          done (2.3s)

Output written to ./output/quizzing-platform/
  prompt_config.json      (42 KB)
  architecture_spec.md    (28 KB)
  scaffolding/            (7 files)

Pipeline complete. Average quality: 8.1/10. Total tokens: 187,432.
```

### Verification

- [ ] `prompter generate "a quizzing platform"` runs the full pipeline end-to-end
- [ ] Progress is displayed in real-time with `rich` spinners/tables
- [ ] Output files are written to `./output/<project_name>/`
- [ ] `prompter interactive "a quizzing platform"` pauses after Analyzer for review
- [ ] `prompter generate --resume <path>` resumes from a checkpoint
- [ ] Pipeline exits with code 0 on success, 1 on failure, 2 on clarification, 3 on config error, 4 on rate limit
- [ ] `--verbose` flag outputs detailed logs to stderr
- [ ] Ctrl+C during execution saves a checkpoint

### Dependencies

Phases 2-6 completed (all agents must be implemented).

---

## Phase 8: Testing + Polish

### Goal

Comprehensive test coverage, error scenario handling, and CLI polish.

### Files to Create / Modify

| File                                      | Purpose                                           |
| ----------------------------------------- | ------------------------------------------------- |
| `tests/unit/test_state.py`                | State creation, serialization, round-trip         |
| `tests/integration/test_analyzer.py`      | Analyzer with mocked LLM                          |
| `tests/integration/test_architect.py`     | Architect with mocked LLM                         |
| `tests/integration/test_critic.py`        | Critic with mocked LLM (high/low scores)          |
| `tests/integration/test_refiner.py`       | Refiner with mocked LLM (revision improves score) |
| `tests/integration/test_packager.py`      | Packager with mocked state                        |
| `tests/integration/test_pipeline.py`      | Full pipeline with mocked LLM                     |
| `tests/regression/test_prompt_quality.py` | Golden-file tests (real API, optional)            |
| `tests/regression/fixtures/`              | Known-good inputs and expected output structures  |

### Test Scenarios

**Unit tests**:

- All Pydantic models: valid data, missing fields, wrong types, boundary values
- Token estimation accuracy
- Config loading from env vars, defaults, missing key error
- State checkpoint serialization round-trip

**Integration tests (mocked LLM)**:

- Analyzer: good input -> valid ModuleMap
- Analyzer: vague input -> `needs_clarification` flag
- Architect: iterates over AI modules, produces one artifact per module
- Critic: assigns low score to weak prompt, high score to strong prompt
- Refiner: produces revised artifact addressing specific feedback
- Packager: assembles valid JSON, Markdown, and scaffolding
- Full pipeline: end-to-end with mocked responses for all agents

**Error scenario tests**:

- Groq API returns 429 -> exponential backoff
- Groq API returns 500 -> 3 retries then fail
- LLM returns malformed JSON -> self-healing retry
- LLM returns valid JSON failing Pydantic -> schema error appended, retry
- Critic loop exhausts max iterations -> best versions used, warnings attached
- Pipeline interruption -> checkpoint saved, resume works

**Regression tests (optional, real API)**:

- Fixed input -> output structure matches golden file (module count, required modules, score ranges)
- Run on schedule, not every commit

### CLI Polish

- `--help` text is clear and complete for all commands and flags
- Error messages are human-readable (not stack traces)
- Missing `GROQ_API_KEY` gives a clear message with instructions
- Invalid input gives specific feedback (too short, too long, etc.)

### Verification

- [ ] `pytest tests/unit/` passes (all unit tests green)
- [ ] `pytest tests/integration/` passes (all integration tests green)
- [ ] `ruff check prompter/` passes (no linting issues)
- [ ] `mypy prompter/` passes (no type errors)
- [ ] Manual end-to-end test: `prompter generate "a quizzing platform for medical students"` produces valid output
- [ ] Manual error test: unset `GROQ_API_KEY`, run `prompter generate "test"` -> clear error message

### Dependencies

Phase 7 completed (full pipeline must be wired).

---

## Phase Dependency Graph

```
Phase 0 ──▶ Phase 1 ──┬──▶ Phase 2 ──▶ Phase 3 ──┬──▶ Phase 5
                       │                            │
                       │                            └──▶ Phase 4
                       │
                       └──▶ Phase 6 (can start models/output early,
                                      needs all agents for full impl)

Phase 2-6 ──▶ Phase 7 ──▶ Phase 8
```

**Critical path**: 0 -> 1 -> 2 -> 3 -> 5 -> 7 -> 8

Phases 4 and 6 can be developed in parallel with other phases once their dependencies are met, but must be complete before Phase 7.

---

## Token Budget for Full Development

Estimated LLM API costs during development and testing:

| Activity                                            | Est. API Calls            | Est. Tokens       |
| --------------------------------------------------- | ------------------------- | ----------------- |
| Manual testing during Phase 2-6 (individual agents) | ~50 calls                 | ~300K tokens      |
| Full pipeline testing during Phase 7                | ~10 full runs x ~30 calls | ~1.5M tokens      |
| Regression test recording                           | ~3 full runs              | ~450K tokens      |
| **Total estimated**                                 |                           | **~2.25M tokens** |

At Groq's free tier or standard pricing, this is manageable. Mocked tests consume zero API tokens.

---

_End of Development Orchestration Plan._
