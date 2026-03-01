"""Module map models — Analyzer output defining project decomposition."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    conversational = "conversational"
    batch = "batch"
    real_time = "real_time"
    adaptive = "adaptive"
    hybrid = "hybrid"


class DomainClassification(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)


class Module(BaseModel):
    name: str
    description: str
    requires_ai: bool
    ai_justification: Optional[str] = None
    ai_capability_needed: Optional[str] = None
    interaction_type: InteractionType
    data_inputs: list[str]
    data_outputs: list[str]
    failure_modes: list[str]
    depends_on: list[str] = Field(default_factory=list)


class ModuleMap(BaseModel):
    project_name: str
    domain_classification: DomainClassification
    interaction_model: InteractionType
    interaction_model_rationale: str
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    modules: list[Module]
    module_count: int
    ai_module_count: int
