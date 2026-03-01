"""Configuration management via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROMPTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- API (GROQ_API_KEY has no prefix — handled via alias) ---
    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
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
    llm_max_tokens: int = 4096

    # --- Rate limiting ---
    rate_limit_tier: str = "free"  # "free" or "paid"
    free_tier_request_delay: float = 5.0  # Minimum seconds between requests (free tier)

    # --- Output ---
    output_dir: str = "./output"
    scaffold_enabled: bool = True
    output_format: str = "both"  # json | markdown | both

    # --- Logging ---
    verbose: bool = False

    def safe_dict(self) -> dict:
        """Return settings as dict with API key excluded (for PipelineState)."""
        d = self.model_dump()
        d.pop("groq_api_key", None)
        return d
