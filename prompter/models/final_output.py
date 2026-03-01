"""Final output models — assembled pipeline deliverable."""

from pydantic import BaseModel

from prompter.models.inter_agent_map import InterAgentMap
from prompter.models.prompt_artifact import PromptArtifact


class PipelineMetadata(BaseModel):
    total_modules: int
    ai_modules: int
    total_estimated_tokens: int
    average_quality_score: float
    critic_iterations_used: int
    total_pipeline_tokens_consumed: int
    generation_duration_seconds: float


class FinalOutputArtifact(BaseModel):
    project: str
    version: str = "1.0"
    generated_at: str
    model_target: str = "llama-3.3-70b-versatile"
    modules: list[PromptArtifact]
    inter_agent_map: InterAgentMap
    pipeline_metadata: PipelineMetadata
