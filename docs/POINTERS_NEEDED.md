# Pointers Needed Before Development

**Status**: ALL RESOLVED
**Date Resolved**: 2026-03-01

---

## Resolved Decisions

### 1. CLI Framework Choice

**Decision**: Typer

Typer is confirmed. Type-hint-based CLI definition, built on Click, aligns with the Pydantic-first philosophy of the project.

---

### 2. Git & Version Control

**Decision**: Fully managed by Claude

All version control is handled autonomously: git init, commits after each phase, pushing to remote, resolving any conflicts. No user intervention required for VCS operations.

---

### 3. Python Version

**Decision**: 3.11+

`requires-python = ">=3.11"` in `pyproject.toml`. Will use 3.11-compatible syntax only (no 3.12+ `type` statement).

---

### 4. System Prompt Authoring

**Decision**: Option A — Full authorship with deep Chain-of-Thought

All 5 agent system prompts will be written in full, acting as a senior system prompt engineer. Each prompt will include:
- Detailed Chain-of-Thought reasoning sections
- Explicit role definition and behavioral boundaries
- Step-by-step processing instructions
- Output schema enforcement
- Edge case handling directives
- Anti-patterns and explicit prohibitions

The prompts are the core IP of the engine and will be treated accordingly.

---

### 5. Groq API Rate Limit Tier

**Decision**: Free tier — design for aggressive throttling

Free tier constraints (~30 requests/min, ~6K tokens/min) require:
- **Request queue** with configurable concurrency (default: 1 concurrent request)
- **Aggressive backoff**: longer base delays (5s between requests minimum)
- **Token-per-minute tracking**: pace requests to stay under ~6K tokens/min
- **Pipeline will be slow** (~5-15 minutes for a full run) but reliable
- **Progress display** must communicate that slowness is expected, not a bug
- Rate limit pre-flight check adjusted for free tier thresholds

The architecture already supports this via the three-layer retry in `llm/client.py`. The `Settings` model will include a `rate_limit_tier` field (`free` | `paid`) that adjusts timing automatically.

---

### 6. Scaffolding Code Style

**Decision**: Async service classes with Pydantic I/O — designed for FastAPI + React reuse

The generated scaffold code will follow a modern async service pattern that can be directly wrapped by FastAPI endpoints and consumed by a React frontend:

```python
from pydantic import BaseModel

class QuestionGeneratorInput(BaseModel):
    topic: str
    difficulty_level: int
    user_history: list[dict] | None = None

class QuestionGeneratorOutput(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str

class QuestionGeneratorAgent:
    """Agent stub for the Question Generator module."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.system_prompt = load_prompt("question_generator")

    async def run(self, input: QuestionGeneratorInput) -> QuestionGeneratorOutput:
        # Implementation here
        ...
```

Key patterns:
- **Pydantic models** for all inputs/outputs (serializable to JSON for REST APIs)
- **Async methods** (FastAPI-native, no synchronous blocking)
- **Dependency injection** via constructor (config, LLM client)
- **Clean separation**: agent logic decoupled from transport layer (CLI, HTTP, WebSocket)
- `scaffolding/main.py` will include both a CLI runner and a comment showing how to mount agents as FastAPI routes

---

### 7. Markdown Spec Audience

**Decision**: Option A — Technical

The generated `architecture_spec.md` will be fully technical:
- Assumes reader understands prompt engineering, LLM concepts, and multi-agent systems
- Includes JSON examples, token budgets, technique rationales
- No dumbed-down summaries or jargon-free sections
- Targets Persona 1 (Solo Developer), Persona 2 (Tech Lead), and Persona 3 (Prompt Engineer)

---

### 8. Interactive Mode Depth

**Decision**: Option B — Full confirmation at every agent stage

Since this is a new system and verifying optimal results is critical:
- Confirmation gate after **Analyzer** (module decomposition review)
- Confirmation gate after **Architect** (review generated prompts per module)
- Confirmation gate after **Communication Designer** (review inter-agent map)
- Confirmation gate after **Critic** (review scores, decide whether to accept or force another refinement round)
- Each gate shows a rich-formatted summary and prompts: `[Y/n/edit]`
- `edit` allows inline adjustments before proceeding
- In `prompter generate` (non-interactive), all gates are auto-accepted

This makes `interactive` mode a proper development/verification tool, not just a thin wrapper.

---

### 9. Progress Display Verbosity

**Decision**: Option B — Moderate

Default (non-verbose) output:
- Per-agent spinner with timing
- Critic score table showing all modules and pass/fail
- Final summary with token usage
- Rate-limit pause indicators (important for free tier — user needs to know it's waiting, not stuck)

---

### 10. Test Coverage Target

**Decision**: Mocked integration tests only

- All tests use mocked LLM responses (zero API calls in test suite)
- Record-and-replay fixtures documented for future regression testing
- No golden-file tests hitting the real Groq API in v1
- Manual end-to-end verification with real API during development

---

### 11. Error Message Style

**Decision**: Layered approach

Three levels of error detail:

| Context | Style | Example |
|---|---|---|
| **User-facing (default)** | Explanatory + actionable | "Pipeline failed at Critic: the LLM returned invalid JSON after 3 attempts. This usually means the model is overloaded. Try again in a few minutes, or run with `--verbose` for details." |
| **Recovery guidance** | Always included | "State saved to `.prompter_state/<run_id>/`. Resume with: `prompter generate --resume <path>`" |
| **Debug (--verbose)** | Full technical detail | Full request/response payloads, Pydantic validation errors, HTTP status codes, retry counts, stack traces |

Design principles:
- Never show raw stack traces in default mode
- Always suggest a next action (retry, simplify input, use --verbose, use --resume)
- Rate limit errors specifically mention the free tier constraints and expected wait time
- Missing `GROQ_API_KEY` gives a clear setup instruction, not a `KeyError`

---

### 12. Output Directory Naming

**Decision**: Option C — Slugified name + timestamp suffix

Output path format: `./output/<slug>_<YYYYMMDD_HHMMSS>/`

Example: `./output/quizzing-platform_20260301_143022/`

This prevents overwrites on repeated runs of the same idea while keeping the directory name human-readable.

---

## Summary (All Resolved)

| #   | Question                 | Decision                                            |
| --- | ------------------------ | --------------------------------------------------- |
| 1   | CLI framework            | **Typer**                                           |
| 2   | Version control          | **Fully managed by Claude** (init, commit, push)    |
| 3   | Python version           | **3.11+**                                           |
| 4   | System prompt authoring  | **Full authorship** with deep CoT, expert role      |
| 5   | Groq rate limit tier     | **Free tier** — aggressive throttling, slow is OK   |
| 6   | Scaffolding code style   | **Async service classes** with Pydantic I/O         |
| 7   | Markdown spec audience   | **Technical** (Persona 1, 2, 3)                     |
| 8   | Interactive mode depth   | **Full** — confirmation gate at every agent stage   |
| 9   | Progress display         | **Moderate** — per-agent spinner, scores, summary   |
| 10  | Test coverage            | **Mocked integration tests only**                   |
| 11  | Error message style      | **Layered** — explanatory default, debug on verbose |
| 12  | Output dir naming        | **Slug + timestamp** (`quizzing-platform_20260301`) |

---

## Impact on Architecture

These decisions require the following updates to `ARCHITECTURE.md` and `ORCHESTRATION_PLAN.md`:

1. **Free tier rate limiting** (Q5): Add request queue with pacing to `llm/client.py`. Add `rate_limit_tier` to `Settings`. Adjust pre-flight estimates for free tier.
2. **Async scaffold pattern** (Q6): Update Packager's scaffold templates. Scaffold must generate `async def` methods and Pydantic I/O models. Include FastAPI integration comment.
3. **Full interactive mode** (Q8): Expand `prompter interactive` to have 4 confirmation gates instead of 1. More complex CLI flow in Phase 7.
4. **Slug + timestamp naming** (Q12): Update output path logic in Packager.

---

_All pointers resolved. Ready for Phase 0._
