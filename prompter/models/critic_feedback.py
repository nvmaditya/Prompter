"""Critic feedback models — quality scoring and issue tracking."""

from enum import Enum

from pydantic import BaseModel, Field


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
    location: str
    description: str
    suggestion: str


class CategoryScore(BaseModel):
    score: float = Field(ge=0, le=10)
    issues: list[Issue] = Field(default_factory=list)


class CriticFeedback(BaseModel):
    module_name: str
    overall_score: float = Field(ge=0, le=10)
    passed: bool
    category_scores: dict[str, CategoryScore]
    issues: list[Issue]
    iteration: int
    summary: str
