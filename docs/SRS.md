# Software Requirement Specification: Prompt Engineering Engine

**Version**: 1.0
**Date**: 2026-03-01
**Status**: Draft
**Author**: Engineering Team

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Data Models and Schemas](#5-data-models-and-schemas)
6. [External Interface Requirements](#6-external-interface-requirements)
7. [Constraints and Assumptions](#7-constraints-and-assumptions)

---

## 1. Introduction

### 1.1 Purpose

This document provides the formal software requirement specification for the Prompt Engineering Engine. It defines all functional and non-functional requirements, data models, external interfaces, and constraints that govern the system's design and implementation. This SRS serves as the binding contract between product requirements (see PRD.md) and technical architecture (see ARCHITECTURE.md).

### 1.2 Scope

This specification covers:

- **Python backend**: All agent logic, LangGraph orchestration, Groq API integration, output generation.
- **CLI interface**: The primary user interface for v1 — command-line tool for submitting project ideas, configuring pipeline parameters, and receiving output artifacts.
- **Output artifact generation**: JSON prompt configuration, Markdown specification, and Python starter code scaffolding.

**Explicitly deferred** from this specification:

- Web UI (React frontend) — to be specified in a future SRS revision.
- Multi-model support — v1 is locked to a single LLM provider and model.
- Authentication, billing, and multi-tenancy.

### 1.3 Definitions and Acronyms

| Term                    | Definition                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CoT**                 | Chain-of-Thought — a prompt engineering technique that instructs the LLM to reason step-by-step before producing a final answer.                                  |
| **Few-Shot**            | A prompt engineering technique that includes example input-output pairs to guide the LLM's output format and behavior.                                            |
| **LangGraph**           | A framework for building stateful, multi-step agent applications as directed graphs with typed state.                                                             |
| **Groq**                | An LLM inference provider offering high-throughput API access to open-source models.                                                                              |
| **Module Map**          | The structured decomposition of a project idea into functional modules, each annotated with AI requirements, data flows, and failure modes.                       |
| **Prompt Architecture** | The complete set of system prompts, agent definitions, inter-agent communication protocols, context management strategies, and evaluation criteria for a project. |
| **Context Slot**        | A `{{variable}}` placeholder in a prompt template that is filled with runtime data when the prompt is executed.                                                   |
| **Shared Memory**       | A state store accessible by multiple agents, used to share data without passing it directly through prompt context.                                               |
| **Handoff Condition**   | A programmable boolean expression that triggers the transfer of control and data from one agent to another.                                                       |
| **Context Pollution**   | When irrelevant or stale information from one agent's context leaks into another agent's input, reducing output quality.                                          |
| **Quality Threshold**   | The minimum Critic score (0-10) a prompt must achieve to pass the Critic/Refiner loop without further refinement. Default: 7.0.                                   |
| **Pipeline**            | The end-to-end execution of all agents in sequence (Analyzer → Architect → Communication Designer → Critic/Refiner → Packager).                                   |
| **Artifact**            | A file produced by the Output Packager: JSON config, Markdown spec, or code scaffolding.                                                                          |

### 1.4 References

| Document                | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `docs/PRD.md`           | Product Requirements Document — product vision, user personas, features, success metrics.          |
| `docs/ARCHITECTURE.md`  | System Architecture Document — technical architecture, LangGraph design, agent prompts, data flow. |
| LangGraph Documentation | https://langchain-ai.github.io/langgraph/                                                          |
| Groq API Documentation  | https://console.groq.com/docs/api-reference                                                        |
| Llama 3.3 Model Card    | Meta's documentation for the llama-3.3-70b-versatile model.                                        |

---

## 2. Overall Description

### 2.1 Product Perspective

The Prompt Engineering Engine is a **standalone, locally-executed system**. It has one external dependency: the Groq API for LLM inference. All other operations (orchestration, validation, output generation) are performed locally.

```
                    ┌─────────────────────────┐
                    │     External: Groq API   │
                    │  llama-3.3-70b-versatile │
                    └────────────┬────────────┘
                                 │ HTTPS
                    ┌────────────▼────────────┐
                    │   Prompt Engineering     │
                    │        Engine            │
                    │                          │
  User ──CLI──►    │  LangGraph Pipeline      │  ──► Output Files
                    │  (5 Agent Nodes)         │      (JSON, MD,
                    │                          │       scaffolding)
                    └─────────────────────────┘
                          Local Machine
```

**System boundary**: The engine consumes the Groq API and produces local file artifacts. It does not expose any network services, does not persist data beyond the current run (unless session history is enabled in a future version), and does not communicate with any service other than Groq.

### 2.2 Product Functions

The system performs the following high-level functions:

1. **Accept a project idea** as natural language text via CLI.
2. **Analyze the project** into functional modules with AI/deterministic classification.
3. **Generate system prompts** for each AI-requiring module using appropriate prompt engineering techniques.
4. **Design inter-agent communication** including shared memory, handoffs, triggers, and pollution prevention.
5. **Critique and refine** all generated prompts through an iterative quality assurance loop.
6. **Package outputs** into structured, deliverable artifacts (JSON, Markdown, code scaffolding).

### 2.3 User Characteristics

The primary users are **CLI-literate developers and technical leads** who:

- Are building AI-powered applications and need prompt architectures.
- Have familiarity with LLM concepts (prompts, tokens, context windows, agents).
- Are comfortable with command-line tools and JSON/Markdown output formats.
- May or may not be expert prompt engineers (the system compensates for this).

### 2.4 Operating Environment

| Requirement      | Specification                                                    |
| ---------------- | ---------------------------------------------------------------- |
| Python           | 3.11 or higher                                                   |
| Operating System | Windows, macOS, or Linux                                         |
| Network          | Internet access required for Groq API calls                      |
| Disk Space       | Minimal — output artifacts are typically <1 MB per run           |
| Memory           | 512 MB minimum (orchestration overhead; LLM inference is remote) |
| API Key          | Valid `GROQ_API_KEY` environment variable                        |

### 2.5 Design and Implementation Constraints

| Constraint                                                    | Rationale                                                                                                                                                                          |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Must use LangGraph** for agent orchestration                | Provides typed state graphs with conditional edges, required for the Critic/Refiner feedback loop. Raw LangChain or custom async loops do not provide the same guarantees.         |
| **Must use Groq's llama-3.3-70b-versatile exclusively**       | No fallback to other providers or models in v1. All prompts are engineered specifically for this model's capabilities and limitations.                                             |
| **Max output per LLM call: 32,768 tokens**                    | Hard limit imposed by Groq API for this model. Affects Output Packager design (may need to split large outputs across multiple calls).                                             |
| **Max context per LLM call: 131,072 tokens**                  | The model's context window. System prompts + user messages + response must fit within this budget.                                                                                 |
| **Rate limits: 300,000 tokens/minute, 1,000 requests/minute** | Groq-imposed limits. The pipeline must respect these through token budget pre-calculation and rate limiting.                                                                       |
| **All prompts designed for Llama 3.3's instruction format**   | Llama 3.3 has specific strengths (IFEval 92.1, strong structured output) and weaknesses (hallucination risk, refusal of benign requests) that inform every prompt design decision. |
| **JSON output mode**                                          | All structured outputs from agents use Groq's `response_format: {"type": "json_object"}` to enforce valid JSON responses.                                                          |

---

## 3. Functional Requirements

### FR-001: Project Idea Input

**Description**: The system accepts a project idea from the user as natural language text.

| ID       | Requirement                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| FR-001.1 | The system SHALL accept a project idea as a string via CLI positional argument: `prompter generate "idea text"`. |
| FR-001.2 | The system SHALL accept a project idea from a file via flag: `prompter generate --file idea.txt`.                |
| FR-001.3 | The system SHALL accept a project idea via interactive prompt in interactive mode: `prompter interactive`.       |
| FR-001.4 | The system SHALL validate that the input is non-empty and contains at least 10 characters.                       |
| FR-001.5 | The system SHALL support multiline input for complex descriptions (via file input or interactive mode).          |
| FR-001.6 | The system SHALL display a confirmation of the received input before proceeding to analysis.                     |
| FR-001.7 | The system SHALL reject inputs exceeding 10,000 characters with an error message suggesting the user summarize.  |

**Acceptance Criteria**: Given a valid project idea (10-10,000 characters), the system acknowledges receipt and proceeds to analysis. Given invalid input, the system displays a clear error message and exits gracefully.

---

### FR-002: Project Analysis (Analyzer Agent)

**Description**: The Analyzer Agent decomposes the project idea into functional modules.

| ID        | Requirement                                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-002.1  | The system SHALL decompose the project idea into 3-15 functional modules.                                                                          |
| FR-002.2  | Each module SHALL include: name, description, AI-vs-deterministic classification, and interaction model (conversational/batch/real_time/adaptive). |
| FR-002.3  | The system SHALL classify the project domain by reasoning about the idea — not from a hardcoded domain list.                                       |
| FR-002.4  | The system SHALL support multi-domain classification (primary + secondary domains).                                                                |
| FR-002.5  | The system SHALL identify AI touchpoints with justification for why AI is needed (vs. deterministic logic).                                        |
| FR-002.6  | The system SHALL infer the overall interaction model (conversational, batch, real_time, adaptive, or hybrid).                                      |
| FR-002.7  | The system SHALL output a structured JSON module map conforming to the `ModuleMap` schema (Section 5.1).                                           |
| FR-002.8  | The system SHALL NOT use template matching, archetype lookup, or any hardcoded project-type classification.                                        |
| FR-002.9  | The system SHALL identify implicit requirements (e.g., a quizzing platform implicitly needs user progress tracking) even if not explicitly stated. |
| FR-002.10 | The system SHALL set a `needs_clarification` flag and list specific questions when the input is too vague to decompose meaningfully.               |
| FR-002.11 | When `needs_clarification` is true, the pipeline SHALL halt and surface the questions to the user before proceeding.                               |

**Acceptance Criteria**: Given "a quizzing platform for medical students with adaptive difficulty", the Analyzer produces a module map with 4-8 modules including at least: question generation, user profiling, adaptive difficulty, and feedback. Domain is classified as Education/Healthcare. Interaction model is adaptive or hybrid.

---

### FR-003: Module Prompt Generation (Architect Agent)

**Description**: The Architect Agent generates a production-quality system prompt for each AI-classified module.

| ID        | Requirement                                                                                                                                               |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-003.1  | The system SHALL generate one system prompt per module classified as requiring AI.                                                                        |
| FR-003.2  | Each prompt SHALL use a technique selected from: Chain-of-Thought, Few-Shot, Role Prompting, Output Constraints, or Socratic Technique.                   |
| FR-003.3  | The system SHALL provide a written rationale for why the selected technique is appropriate for the module.                                                |
| FR-003.4  | Up to two techniques MAY be combined per module, with one designated as primary.                                                                          |
| FR-003.5  | Each prompt SHALL include `{{variable}}` context slots for dynamic runtime data injection.                                                                |
| FR-003.6  | Every context slot SHALL be documented with: what fills the slot, when it is injected, fallback value if missing, and whether it is required or optional. |
| FR-003.7  | Each prompt SHALL include a token estimate broken down as: system prompt tokens, expected context tokens, and expected output tokens.                     |
| FR-003.8  | Generated prompts SHALL be self-contained — a reader should understand the agent's full purpose from the prompt alone.                                    |
| FR-003.9  | Generated prompts SHALL NOT exceed 1,500 words.                                                                                                           |
| FR-003.10 | Each generated prompt SHALL include at least 2 explicit constraints or boundaries.                                                                        |
| FR-003.11 | Each generated prompt SHALL include an output format specification (preferably a JSON schema).                                                            |

**Acceptance Criteria**: For a "Question Generation Module" classified as Few-Shot + Output Constraints, the Architect produces a prompt that includes: role definition, task description, context slots (e.g., `{{topic}}`, `{{difficulty_level}}`), numbered processing steps, JSON output schema, constraints, and edge case handling. Token estimate is provided.

---

### FR-004: Inter-Agent Communication Design (Communication Designer Agent)

**Description**: The Communication Designer produces the communication infrastructure between AI modules.

| ID       | Requirement                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-004.1 | The system SHALL produce a shared memory schema defining all cross-agent data structures.                                                                                             |
| FR-004.2 | Each shared memory field SHALL document: type, which agent(s) write to it, which read from it, when it is updated, and its default value.                                             |
| FR-004.3 | The system SHALL define handoff conditions between agents: trigger condition (as a programmable boolean), data passed, format, and fallback if data is incomplete.                    |
| FR-004.4 | The system SHALL define context pollution prevention rules specifying: what data is protected, which agents must not see it, the risk if they do, and how to enforce the restriction. |
| FR-004.5 | The system SHALL produce a trigger map: event name, which agent(s) it activates, priority order, execution mode (sequential or parallel), and error fallback.                         |
| FR-004.6 | Every `data_inputs` and `data_outputs` from the module map SHALL be represented in either a shared memory field or a handoff condition — no implicit data passing.                    |
| FR-004.7 | Context pollution prevention rules SHALL address at minimum: prompt text isolation between agents, PII scope limitation, and error state propagation prevention.                      |

**Acceptance Criteria**: For a 5-module quizzing platform, the Communication Designer produces: a shared memory schema with at least 3 fields, handoff conditions for each agent transition, at least 2 pollution prevention rules, and a trigger map covering both normal flow and error cases.

---

### FR-005: Prompt Critique (Critic Agent)

**Description**: The Critic Agent evaluates each generated prompt for quality across multiple dimensions.

| ID       | Requirement                                                                                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-005.1 | The system SHALL evaluate each generated prompt against five categories: ambiguity, hallucination risk, missing constraints, edge case coverage, and token efficiency.                    |
| FR-005.2 | Each category SHALL be scored on a 0-10 scale.                                                                                                                                            |
| FR-005.3 | The overall quality score SHALL be a weighted average: Ambiguity 25%, Hallucination Risk 25%, Missing Constraints 20%, Edge Cases 15%, Token Efficiency 15%.                              |
| FR-005.4 | Each issue flagged SHALL include: category, severity (low/medium/high/critical), location (quoted text from the prompt), description, and a specific actionable suggestion for fixing it. |
| FR-005.5 | The system SHALL flag any prompt with an overall score below the configurable quality threshold (default: 7.0/10).                                                                        |
| FR-005.6 | The system SHALL flag a prompt as `passed: false` if any individual category scores below a configurable per-dimension threshold.                                                         |
| FR-005.7 | The Critic SHALL produce output conforming to the `CriticFeedback` schema (Section 5.4).                                                                                                  |

**Acceptance Criteria**: Given a prompt with vague instructions and no edge case handling, the Critic scores it below 7.0, flags specific ambiguous phrases with quoted text, and provides concrete rewrite suggestions.

---

### FR-006: Prompt Refinement (Refiner)

**Description**: The Refiner re-generates prompts that fail the Critic's quality threshold.

| ID       | Requirement                                                                                                                                                                  |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-006.1 | The system SHALL re-generate any prompt that falls below the quality threshold, incorporating the Critic's specific feedback.                                                |
| FR-006.2 | The maximum number of Critic/Refiner iterations SHALL be configurable (default: 3).                                                                                          |
| FR-006.3 | The loop SHALL terminate when ALL prompts pass the quality threshold OR the maximum iteration count is reached.                                                              |
| FR-006.4 | The system SHALL preserve the full iteration history: original prompt, each Critic evaluation, each Refiner revision.                                                        |
| FR-006.5 | When maximum iterations are reached without all prompts passing, the system SHALL use the highest-scoring version of each prompt and attach a quality warning to the output. |
| FR-006.6 | The Refiner SHALL receive both the original prompt and the Critic's feedback as input — not just the feedback alone.                                                         |

**Acceptance Criteria**: A prompt scoring 5.5/10 on the first pass receives Critic feedback, is revised by the Refiner, re-evaluated at 7.8/10 on the second pass, and passes. The iteration history records both versions.

---

### FR-007: Output Packaging (Packager Agent)

**Description**: The Packager Agent assembles all pipeline outputs into deliverable artifacts.

| ID       | Requirement                                                                                                                                                                                                                        |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-007.1 | The system SHALL produce a JSON prompt configuration file conforming to the `FinalOutputArtifact` schema (Section 5.5).                                                                                                            |
| FR-007.2 | The system SHALL produce a Markdown specification document that is standalone-readable without the JSON file.                                                                                                                      |
| FR-007.3 | The system SHALL optionally produce starter code scaffolding: Python files with prompt templates, agent stubs, and a README.                                                                                                       |
| FR-007.4 | All output files SHALL be written to a configurable output directory (default: `./output/<project_name>/`).                                                                                                                        |
| FR-007.5 | The JSON configuration SHALL include quality scores for each module. Modules with scores below 7.0 SHALL be flagged with a visible warning.                                                                                        |
| FR-007.6 | The Markdown specification SHALL include: executive summary, module-by-module breakdown with technique rationale, inter-agent communication overview, quality assessment summary, implementation notes, and token budget analysis. |
| FR-007.7 | The Packager SHALL NOT modify prompt text from the Architect/Refiner — it packages exactly what was produced.                                                                                                                      |

**Acceptance Criteria**: A completed pipeline produces three files in `./output/quizzing-platform/`: `prompt_config.json` (valid, parseable JSON), `architecture_spec.md` (readable Markdown), and a `scaffolding/` directory with Python stubs.

---

### FR-008: Prompt Chaining Logic

**Description**: The system designs conditional sequences between prompts within the generated architecture.

| ID       | Requirement                                                                                                                                                                 |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-008.1 | The system SHALL define conditional prompt chains: if condition X is met by a prompt's output, then prompt Y fires next.                                                    |
| FR-008.2 | Each chain SHALL document: the sequence of steps, branching conditions, data passed between steps, and error/fallback behavior if a step fails.                             |
| FR-008.3 | Chains SHALL be represented in the JSON config as directed graphs with annotated edges (conditions and data).                                                               |
| FR-008.4 | Each prompt artifact SHALL document its position in any chain: what fires before it, what triggers it, where its output goes, and what conditional branches exist after it. |

**Acceptance Criteria**: For a quizzing platform, the system produces a chain: User Understanding → (if engagement low) Motivation Module → Question Generation → (if score < 60%) Remediation Module → Feedback Module. Each transition has a condition and data specification.

---

### FR-009: Token Budget Management

**Description**: The system estimates and manages token consumption across the generated prompt architecture.

| ID       | Requirement                                                                                                                                                                                               |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-009.1 | The system SHALL estimate tokens for each generated prompt: system prompt tokens + expected context slot tokens + expected output tokens.                                                                 |
| FR-009.2 | The system SHALL flag any single prompt exceeding a configurable per-prompt threshold (default: 8,000 tokens total).                                                                                      |
| FR-009.3 | The system SHALL calculate and flag total pipeline token consumption exceeding a configurable pipeline threshold (default: 50,000 tokens).                                                                |
| FR-009.4 | When a prompt or pipeline exceeds its threshold, the system SHALL provide specific compression suggestions (e.g., "Remove 2 of 4 few-shot examples to save ~200 tokens" — not generic "make it shorter"). |
| FR-009.5 | Token estimates SHALL be included in both the JSON config and the Markdown spec.                                                                                                                          |

**Acceptance Criteria**: A 7-module architecture shows per-module token estimates and a pipeline total. One module with 4 few-shot examples is flagged as over-budget with a specific suggestion to reduce examples.

---

### FR-010: Evaluation Criteria Generation

**Description**: The system generates testing and evaluation criteria for each generated prompt.

| ID       | Requirement                                                                                                                       |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| FR-010.1 | Each prompt SHALL include at least 2 positive output examples (what good output looks like).                                      |
| FR-010.2 | Each prompt SHALL include at least 2 negative output examples (what bad output looks like).                                       |
| FR-010.3 | Each prompt SHALL include at least 3 automated evaluation suggestions (e.g., "Assert output JSON contains key 'recommendation'"). |
| FR-010.4 | Each prompt SHALL include human review criteria (what a human evaluator should look for).                                         |
| FR-010.5 | Examples and criteria SHALL be specific to the module's domain and function — not generic.                                        |
| FR-010.6 | Evaluation criteria SHALL be included in both the JSON config (under `eval_criteria`) and the Markdown spec.                      |

**Acceptance Criteria**: A "Question Generation Module" includes: (good) "Generates a question directly relevant to cardiology at difficulty 3", (bad) "Generates a question about an unrelated topic", and automated checks like "Assert output contains 'question' and 'options' keys with correctly typed values".

---

## 4. Non-Functional Requirements

### NFR-001: Performance

| ID        | Requirement                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| NFR-001.1 | The full pipeline SHALL complete in under 5 minutes for a typical project idea (5-10 modules).                |
| NFR-001.2 | Individual agent LLM calls SHALL timeout after 60 seconds, with retry logic before failure.                   |
| NFR-001.3 | The CLI SHALL display progress indicators showing which agent is currently active and overall pipeline stage. |

### NFR-002: Reliability

| ID        | Requirement                                                                                                                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-002.1 | The system SHALL retry failed Groq API calls up to 3 times with exponential backoff (1s, 2s, 4s delays).                                                                                                                |
| NFR-002.2 | The system SHALL handle HTTP 429 (rate limit) responses by waiting for the `retry-after` header duration before retrying.                                                                                               |
| NFR-002.3 | The system SHALL save intermediate pipeline state after each agent completes, enabling resumption if the pipeline fails mid-execution.                                                                                  |
| NFR-002.4 | The system SHALL validate all LLM responses against Pydantic schemas before passing them to the next agent. Schema validation failures trigger a retry with the validation error appended to the prompt (self-healing). |

### NFR-003: Token Efficiency

| ID        | Requirement                                                                                                                       |
| --------- | --------------------------------------------------------------------------------------------------------------------------------- |
| NFR-003.1 | The engine's own agent system prompts (used internally by the pipeline) SHALL each be under 2,000 tokens.                         |
| NFR-003.2 | The system SHALL track cumulative token usage across all LLM calls in a pipeline run and report the total at pipeline completion. |
| NFR-003.3 | The system SHALL display per-agent token usage in verbose mode.                                                                   |

### NFR-004: Extensibility

| ID        | Requirement                                                                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NFR-004.1 | Adding a new agent to the pipeline SHALL require only: (a) defining the agent function, (b) adding a node to the LangGraph graph, and (c) defining edges to/from the new node.       |
| NFR-004.2 | Prompt engineering techniques SHALL be pluggable — new techniques can be added without modifying the Architect Agent's core code. Techniques are registered in a technique registry. |
| NFR-004.3 | Output formatters SHALL be pluggable — new output formats (e.g., YAML, PDF) can be added without modifying the Packager Agent's core code.                                           |

### NFR-005: Security

| ID        | Requirement                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------- |
| NFR-005.1 | API keys SHALL be loaded exclusively from environment variables. No hardcoded keys in source code or configuration files. |
| NFR-005.2 | Generated outputs SHALL NOT contain or leak the engine's own internal system prompts.                                     |
| NFR-005.3 | The system SHALL NOT log or persist API keys in any log file or intermediate state file.                                  |
| NFR-005.4 | The `.env` file SHALL be listed in `.gitignore`.                                                                          |

### NFR-006: Observability

| ID        | Requirement                                                                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-006.1 | The system SHALL log each agent invocation with: agent name, input token count, output token count, latency (ms), and success/failure status.                                          |
| NFR-006.2 | The system SHALL support a `--verbose` flag that outputs detailed logs to stderr including: full LLM request/response payloads, schema validation results, and Critic scoring details. |
| NFR-006.3 | The system SHALL produce a pipeline summary at completion: total duration, total tokens used, per-agent breakdown, final quality scores, and warnings.                                 |

---

## 5. Data Models and Schemas

All data models are defined as Pydantic v2 `BaseModel` classes. These schemas serve as both runtime validation and documentation of the data contracts between agents.

### 5.1 ModuleMap

Output of the Analyzer Agent (FR-002).

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class InteractionType(str, Enum):
    conversational = "conversational"
    batch = "batch"
    real_time = "real_time"
    adaptive = "adaptive"
    hybrid = "hybrid"

class DomainClassification(BaseModel):
    primary: str = Field(description="Primary domain of the project")
    secondary: list[str] = Field(default_factory=list, description="Secondary domains if applicable")

class Module(BaseModel):
    name: str = Field(description="Unique, descriptive module name")
    description: str = Field(description="1-2 sentence description of what the module does")
    requires_ai: bool = Field(description="Whether this module needs LLM reasoning")
    ai_justification: Optional[str] = Field(default=None, description="Why AI is needed (null if requires_ai is false)")
    ai_capability_needed: Optional[str] = Field(default=None, description="generation/analysis/classification/conversation/summarization/transformation")
    interaction_type: InteractionType = Field(description="How this module interacts with its inputs")
    data_inputs: list[str] = Field(description="Data flowing into this module")
    data_outputs: list[str] = Field(description="Data flowing out of this module")
    failure_modes: list[str] = Field(description="What happens if this module fails or performs poorly")
    depends_on: list[str] = Field(default_factory=list, description="Module names this module depends on")

class ModuleMap(BaseModel):
    project_name: str = Field(description="Concise project name derived from the idea")
    domain_classification: DomainClassification
    interaction_model: InteractionType
    interaction_model_rationale: str = Field(description="Why this interaction model was chosen")
    needs_clarification: bool = Field(default=False)
    clarification_questions: list[str] = Field(default_factory=list)
    modules: list[Module]
    module_count: int = Field(description="Must match len(modules)")
    ai_module_count: int = Field(description="Count of modules where requires_ai is true")
```

### 5.2 PromptArtifact

Output of the Architect Agent per module (FR-003).

```python
class ContextSlot(BaseModel):
    variable: str = Field(description="The {{variable}} name as it appears in the prompt")
    description: str = Field(description="What runtime data fills this slot")
    source: str = Field(description="Where/how this data becomes available")
    injection_time: str = Field(description="When this data is injected (e.g., 'session_start', 'post_quiz')")
    fallback: str = Field(description="What to use if this data is missing")
    required: bool = Field(description="Whether the prompt can function without this slot")

class TokenEstimate(BaseModel):
    system_tokens: int = Field(description="Tokens in the system prompt itself")
    expected_context_tokens: int = Field(description="Estimated tokens from all context slots when filled")
    expected_output_tokens: int = Field(description="Estimated tokens in the LLM's response")
    total: int = Field(description="Sum of above three fields")
    budget_warning: Optional[str] = Field(default=None, description="Warning if total exceeds thresholds")

class ChainCondition(BaseModel):
    condition: str = Field(description="Boolean expression on the prompt's output (e.g., 'engagement_level == low')")
    next_prompt: str = Field(description="Module name of the next prompt in the chain")
    context_passed: list[str] = Field(description="Which output fields are passed to the next prompt")

class PromptChain(BaseModel):
    conditions: list[ChainCondition]

class EvalCriteria(BaseModel):
    good_output_examples: list[str] = Field(min_length=2)
    bad_output_examples: list[str] = Field(min_length=2)
    automated_eval_suggestions: list[str] = Field(min_length=3)
    human_review_criteria: list[str] = Field(min_length=1)

class PromptArtifact(BaseModel):
    module_name: str
    agent_role: str = Field(description="The persona/role of this agent (e.g., 'Learner Profiler')")
    primary_technique: str = Field(description="Primary prompt engineering technique used")
    secondary_technique: Optional[str] = Field(default=None)
    technique_rationale: str = Field(description="Why these techniques were chosen for this module")
    system_prompt: str = Field(description="The complete system prompt text")
    context_slots: list[ContextSlot]
    token_estimate: TokenEstimate
    triggers: list[str] = Field(description="Events that cause this prompt to execute")
    outputs_to: list[str] = Field(description="Module names that receive this prompt's output")
    prompt_chain: Optional[PromptChain] = Field(default=None)
    eval_criteria: EvalCriteria
```

### 5.3 InterAgentMap

Output of the Communication Designer Agent (FR-004).

```python
class SharedMemoryField(BaseModel):
    type: str = Field(description="string | number | boolean | object | array")
    description: str
    written_by: list[str] = Field(description="Agent names that write to this field")
    read_by: list[str] = Field(description="Agent names that read from this field")
    updated_on: str = Field(description="Event or condition that triggers an update")
    default: str = Field(description="Default value as a string representation")

class HandoffCondition(BaseModel):
    from_agent: str
    to_agent: str
    condition: str = Field(description="Programmable boolean expression triggering the handoff")
    data_passed: dict[str, str] = Field(description="Mapping of field name to description of data passed")
    format: str = Field(description="Format of the passed data (e.g., 'JSON object', 'string')")
    fallback_if_incomplete: str = Field(description="What happens if the handoff data is missing or partial")

class ContextPollutionRule(BaseModel):
    protected_data: str = Field(description="What data or context is being protected")
    excluded_agents: list[str] = Field(description="Which agents must NOT see this data")
    risk: str = Field(description="What could go wrong if the rule is violated")
    enforcement: str = Field(description="How this rule is enforced (filtering, scoping, namespacing)")

class Trigger(BaseModel):
    event: str = Field(description="Event name and description")
    activates: list[str] = Field(description="Agent names activated by this event")
    priority_order: list[str] = Field(description="Order of activation if multiple agents are triggered")
    execution: str = Field(description="'sequential' or 'parallel'")
    error_fallback: str = Field(description="What happens if an activated agent fails")

class InterAgentMap(BaseModel):
    shared_memory_schema: dict[str, SharedMemoryField]
    handoff_conditions: list[HandoffCondition]
    context_pollution_rules: list[ContextPollutionRule]
    trigger_map: list[Trigger]
```

### 5.4 CriticFeedback

Output of the Critic Agent per prompt (FR-005).

```python
class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class IssueCategory(str, Enum):
    ambiguity = "ambiguity"
    hallucination_risk = "hallucination_risk"
    missing_constraints = "missing_constraints"
    edge_cases = "edge_cases"
    token_efficiency = "token_efficiency"

class Issue(BaseModel):
    category: IssueCategory
    severity: Severity
    location: str = Field(description="Quoted text from the prompt that has the issue")
    description: str = Field(description="What the problem is")
    suggestion: str = Field(description="Specific, actionable fix — not generic advice")

class CategoryScore(BaseModel):
    score: float = Field(ge=0, le=10)
    issues: list[Issue] = Field(default_factory=list)

class CriticFeedback(BaseModel):
    module_name: str
    overall_score: float = Field(ge=0, le=10, description="Weighted average of category scores")
    passed: bool = Field(description="True if overall_score >= quality_threshold")
    category_scores: dict[str, CategoryScore] = Field(
        description="Keys: ambiguity, hallucination_risk, missing_constraints, edge_cases, token_efficiency"
    )
    issues: list[Issue] = Field(description="Flat list of all issues across categories")
    iteration: int = Field(description="Which Critic/Refiner iteration produced this feedback")
    summary: str = Field(description="2-3 sentence overall assessment")
```

### 5.5 FinalOutputArtifact

The complete JSON output (FR-007). This is what gets written to `prompt_config.json`.

```python
class PipelineMetadata(BaseModel):
    total_modules: int
    ai_modules: int
    total_estimated_tokens: int
    average_quality_score: float
    critic_iterations_used: int
    total_pipeline_tokens_consumed: int = Field(description="Actual tokens used by the engine during generation")
    generation_duration_seconds: float

class FinalOutputArtifact(BaseModel):
    project: str
    version: str = Field(default="1.0")
    generated_at: str = Field(description="ISO 8601 timestamp")
    model_target: str = Field(default="llama-3.3-70b-versatile")
    modules: list[PromptArtifact]
    inter_agent_map: InterAgentMap
    pipeline_metadata: PipelineMetadata
```

---

## 6. External Interface Requirements

### 6.1 Groq API Interface

| Property             | Specification                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| **Endpoint**         | `https://api.groq.com/openai/v1/chat/completions`                                                         |
| **Protocol**         | HTTPS (OpenAI-compatible chat completions API)                                                            |
| **Model**            | `llama-3.3-70b-versatile`                                                                                 |
| **Authentication**   | Bearer token via `GROQ_API_KEY` environment variable                                                      |
| **Request Format**   | JSON body with `model`, `messages` (system + user), `response_format`, `max_tokens`, `temperature`        |
| **Response Parsing** | Extract `choices[0].message.content`, parse as JSON when structured output is expected                    |
| **JSON Mode**        | Enabled via `response_format: {"type": "json_object"}` for all structured agent outputs                   |
| **Temperature**      | `0.7` for creative agents (Analyzer, Architect), `0.3` for analytical agents (Critic), `0.5` for Packager |
| **Max Tokens**       | Set per-agent based on expected output size; never exceeds 32,768                                         |

### 6.2 CLI Interface

**Commands**:

```
prompter generate <idea>              # Generate from inline text
prompter generate --file <path>       # Generate from file
prompter interactive                  # Step-by-step with confirmations
prompter --version                    # Show version
prompter --help                       # Show help
```

**Flags** (applicable to `generate` and `interactive`):

| Flag                  | Type    | Default                    | Description                                             |
| --------------------- | ------- | -------------------------- | ------------------------------------------------------- |
| `--output-dir`        | path    | `./output/<project_name>/` | Where to write output artifacts                         |
| `--verbose`           | boolean | false                      | Enable detailed logging to stderr                       |
| `--max-iterations`    | int     | 3                          | Maximum Critic/Refiner loop iterations                  |
| `--quality-threshold` | float   | 7.0                        | Minimum overall quality score to pass Critic            |
| `--format`            | choice  | `both`                     | Output format: `json`, `markdown`, or `both`            |
| `--scaffold`          | boolean | true                       | Whether to generate starter code scaffolding            |
| `--resume`            | path    | none                       | Path to intermediate state file for pipeline resumption |

**Exit Codes**:

| Code | Meaning                                                        |
| ---- | -------------------------------------------------------------- |
| 0    | Success — all artifacts generated                              |
| 1    | Pipeline error — agent failure after retries exhausted         |
| 2    | Input validation error — invalid or empty input                |
| 3    | Configuration error — missing API key or invalid config        |
| 4    | Rate limit error — Groq API rate limits exceeded after retries |

### 6.3 File System Interface

**Output Directory Structure**:

```
output/
  <project_name>/
    prompt_config.json          # FinalOutputArtifact JSON
    architecture_spec.md        # Markdown specification document
    scaffolding/
      README.md                 # Setup instructions
      prompts/
        <module_name>.txt       # Individual prompt files (one per module)
      agents/
        <module_name>_agent.py  # Agent stub files (one per AI module)
      config.py                 # Configuration template
      main.py                   # Entry point template
```

**Intermediate State Files** (for pipeline resumption):

```
.prompter_state/
  <run_id>/
    pipeline_state.json         # Full PipelineState snapshot after each agent
    agent_logs.jsonl            # Per-agent invocation logs
```

---

## 7. Constraints and Assumptions

### 7.1 Constraints

| ID    | Constraint                                                                                                                                                                  | Impact                                                                                                                                                                        |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C-001 | Groq API must remain available at the documented endpoint with the current pricing model.                                                                                   | If Groq changes its API or discontinues the model, the engine requires updates to the client wrapper.                                                                         |
| C-002 | The 32,768 max output token limit means projects with 15 modules may require the Packager to split its work across multiple LLM calls.                                      | Packager must detect truncation and implement multi-call assembly.                                                                                                            |
| C-003 | Token estimation uses `tiktoken` (cl100k_base encoding) as an approximation, introducing ~5-10% variance from Llama 3.3's actual tokenizer.                                 | Token budget warnings may trigger slightly early or late. For precise counting, the `transformers` library's Llama tokenizer can be used at the cost of a heavier dependency. |
| C-004 | The system is single-user, single-run. No concurrent pipeline executions are supported in v1.                                                                               | No need for locking, queuing, or multi-tenant isolation.                                                                                                                      |
| C-005 | All LLM calls are synchronous from the pipeline's perspective (one agent completes before the next starts), except for the Architect which MAY process modules in parallel. | Pipeline latency is dominated by sequential LLM call latency.                                                                                                                 |

### 7.2 Assumptions

| ID    | Assumption                                                                                                                             | Risk if Invalid                                                                                                                                                           |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-001 | Users have a valid Groq API key with sufficient quota for at least one full pipeline run (~15-20 LLM calls, ~150K-200K tokens).        | Pipeline fails at the first LLM call with an authentication or quota error.                                                                                               |
| A-002 | The project idea provided by the user describes a software system that has genuine AI touchpoints (not a purely deterministic system). | The Analyzer may produce a module map with zero AI modules, resulting in no prompts to generate. The pipeline handles this by warning the user.                           |
| A-003 | Network connectivity to `api.groq.com` is stable.                                                                                      | Transient network failures are handled by retry logic. Prolonged outages cause pipeline failure.                                                                          |
| A-004 | Llama 3.3's behavior remains consistent across Groq API versions.                                                                      | If Groq updates to a different Llama checkpoint or modifies inference parameters, prompt behavior may change. Prompt regression tests (see Architecture doc) detect this. |
| A-005 | Users can install Python 3.11+ and pip dependencies on their machine.                                                                  | The engine cannot run if the environment is not set up. Installation instructions will be provided.                                                                       |

---

_End of Software Requirement Specification._
