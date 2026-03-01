"""Prompt artifact models — Architect output with system prompts and metadata."""

from typing import Optional

from pydantic import BaseModel, Field


class ContextSlot(BaseModel):
    variable: str
    description: str
    source: str
    injection_time: str
    fallback: str
    required: bool


class TokenEstimate(BaseModel):
    system_tokens: int
    expected_context_tokens: int
    expected_output_tokens: int
    total: int
    budget_warning: Optional[str] = None


class ChainCondition(BaseModel):
    condition: str
    next_prompt: str
    context_passed: list[str]


class PromptChain(BaseModel):
    conditions: list[ChainCondition]


class EvalCriteria(BaseModel):
    good_output_examples: list[str] = Field(min_length=2)
    bad_output_examples: list[str] = Field(min_length=2)
    automated_eval_suggestions: list[str] = Field(min_length=3)
    human_review_criteria: list[str] = Field(min_length=1)


class PromptArtifact(BaseModel):
    module_name: str
    agent_role: str
    primary_technique: str
    secondary_technique: Optional[str] = None
    technique_rationale: str
    system_prompt: str
    context_slots: list[ContextSlot]
    token_estimate: TokenEstimate
    triggers: list[str]
    outputs_to: list[str]
    prompt_chain: Optional[PromptChain] = None
    eval_criteria: EvalCriteria
