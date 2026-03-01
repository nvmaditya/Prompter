"""LLM client with three-layer retry: HTTP transport, rate limiting, schema self-healing."""

import json
import logging
import time
from typing import Optional

from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from prompter.config import Settings

logger = logging.getLogger(__name__)

# Module-level rate limiter state
_last_request_time: float = 0.0


class LLMError(Exception):
    """Base error for LLM client failures."""


class SchemaValidationError(LLMError):
    """Raised when LLM output fails Pydantic validation after all retries."""


class RateLimitError(LLMError):
    """Raised when rate limit is exhausted after all retries."""


def _get_client(settings: Settings, client: Optional[ChatGroq] = None) -> ChatGroq:
    """Get or create a ChatGroq client."""
    if client is not None:
        return client
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=settings.creative_temperature,
        max_tokens=4096,
        timeout=settings.llm_timeout_seconds,
    )


def _enforce_rate_limit(settings: Settings) -> None:
    """Enforce minimum delay between requests for free tier."""
    global _last_request_time
    if settings.rate_limit_tier == "free" and _last_request_time > 0:
        elapsed = time.time() - _last_request_time
        wait_time = settings.free_tier_request_delay - elapsed
        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s before next request")
            time.sleep(wait_time)
    _last_request_time = time.time()


def _invoke_llm(
    client: ChatGroq,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Make a single LLM call and return raw content string."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    # Override temperature and max_tokens per-call
    response = client.invoke(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content


def _invoke_with_correction(
    client: ChatGroq,
    system_prompt: str,
    user_message: str,
    correction_message: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Re-invoke LLM with original messages plus a correction message."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
        AIMessage(content="I'll fix the issues and provide valid JSON."),
        HumanMessage(content=correction_message),
    ]
    response = client.invoke(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content


def _extract_json(raw: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove first line (```json or ```) and last line (```)
        json_lines = []
        started = False
        for line in lines:
            if not started and line.strip().startswith("```"):
                started = True
                continue
            if started and line.strip() == "```":
                break
            if started:
                json_lines.append(line)
        return "\n".join(json_lines)
    return stripped


def call_llm(
    system_prompt: str,
    user_message: str,
    response_model: type[BaseModel],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    settings: Optional[Settings] = None,
    client: Optional[ChatGroq] = None,
) -> BaseModel:
    """Call the Groq LLM with three-layer retry and return a validated Pydantic model.

    Retry layers:
        1. HTTP transport: 3 retries with exponential backoff (tenacity)
        2. Rate limiting: waits for free-tier pacing
        3. Schema self-healing: re-prompts with validation errors (up to schema_retry_limit)

    Args:
        system_prompt: The system prompt text.
        user_message: The user message text.
        response_model: Pydantic model class to validate the response against.
        temperature: LLM temperature for this call.
        max_tokens: Maximum response tokens.
        settings: Settings instance. Created from env if not provided.
        client: Optional pre-built ChatGroq client (for testing).

    Returns:
        Validated Pydantic model instance.

    Raises:
        SchemaValidationError: After exhausting schema retries.
        LLMError: On unrecoverable LLM errors.
    """
    if settings is None:
        settings = Settings()

    llm_client = _get_client(settings, client)
    schema_retry_limit = settings.schema_retry_limit

    # Append JSON format instruction to system prompt
    json_schema = json.dumps(response_model.model_json_schema(), indent=2)
    augmented_system = (
        f"{system_prompt}\n\n"
        f"You MUST respond with valid JSON matching this schema:\n"
        f"```json\n{json_schema}\n```\n"
        f"Respond ONLY with the JSON object, no additional text."
    )

    # Layer 1: HTTP transport retry via tenacity
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential_jitter(initial=settings.retry_base_delay, max=30),
        reraise=True,
    )
    def _call_with_http_retry(sys_prompt: str, usr_msg: str, correction: Optional[str] = None) -> str:
        # Layer 2: Rate limiting
        _enforce_rate_limit(settings)

        if correction:
            return _invoke_with_correction(
                llm_client, sys_prompt, usr_msg, correction, temperature, max_tokens
            )
        return _invoke_llm(llm_client, sys_prompt, usr_msg, temperature, max_tokens)

    # First attempt
    raw = _call_with_http_retry(augmented_system, user_message)
    json_str = _extract_json(raw)

    # Layer 3: Schema self-healing
    for attempt in range(schema_retry_limit + 1):
        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt >= schema_retry_limit:
                raise SchemaValidationError(
                    f"Failed to get valid {response_model.__name__} after "
                    f"{schema_retry_limit + 1} attempts. Last error: {e}"
                ) from e

            # Build correction prompt
            error_detail = str(e)
            correction = (
                f"Your previous response had validation errors:\n\n{error_detail}\n\n"
                f"The expected JSON schema is:\n```json\n{json_schema}\n```\n\n"
                f"Please provide a corrected JSON response matching the schema exactly."
            )
            logger.warning(
                f"Schema validation failed (attempt {attempt + 1}/{schema_retry_limit + 1}): {e}"
            )

            raw = _call_with_http_retry(augmented_system, user_message, correction)
            json_str = _extract_json(raw)

    # Should not reach here, but just in case
    raise SchemaValidationError(f"Exhausted all retries for {response_model.__name__}")
