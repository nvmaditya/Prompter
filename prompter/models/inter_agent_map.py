"""Inter-agent communication models — Communication Designer output."""

from pydantic import BaseModel


class SharedMemoryField(BaseModel):
    type: str
    description: str
    written_by: list[str]
    read_by: list[str]
    updated_on: str
    default: str


class HandoffCondition(BaseModel):
    from_agent: str
    to_agent: str
    condition: str
    data_passed: dict[str, str]
    format: str
    fallback_if_incomplete: str


class ContextPollutionRule(BaseModel):
    protected_data: str
    excluded_agents: list[str]
    risk: str
    enforcement: str


class Trigger(BaseModel):
    event: str
    activates: list[str]
    priority_order: list[str]
    execution: str
    error_fallback: str


class InterAgentMap(BaseModel):
    shared_memory_schema: dict[str, SharedMemoryField]
    handoff_conditions: list[HandoffCondition]
    context_pollution_rules: list[ContextPollutionRule]
    trigger_map: list[Trigger]
