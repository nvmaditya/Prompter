"""Token estimation using tiktoken."""

import tiktoken


def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
    """Estimate the number of tokens in a text string.

    Uses tiktoken with the cl100k_base encoding (used by GPT-4, also a
    reasonable approximation for Llama models).

    Args:
        text: The text to estimate tokens for.
        model: The tiktoken encoding name.

    Returns:
        Estimated token count.
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))
