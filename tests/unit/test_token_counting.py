"""Unit tests for token counting utility."""

from prompter.utils.tokens import estimate_tokens


class TestEstimateTokens:
    def test_simple_text(self):
        count = estimate_tokens("hello world")
        assert 1 <= count <= 5

    def test_empty_string(self):
        count = estimate_tokens("")
        assert count == 0

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = estimate_tokens(text)
        # ~100 words, roughly 100-150 tokens
        assert 50 <= count <= 200

    def test_json_text(self):
        text = '{"name": "test", "value": 42, "items": ["a", "b", "c"]}'
        count = estimate_tokens(text)
        assert count > 0

    def test_code_text(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        count = estimate_tokens(code)
        assert 20 <= count <= 60

    def test_system_prompt_size(self):
        # A typical system prompt might be ~500-2000 tokens
        prompt = "You are an expert system prompt engineer. " * 100
        count = estimate_tokens(prompt)
        assert count > 200
