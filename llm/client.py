"""LLM client factory."""

import os

from llm.adapters.base import BaseLLMAdapter
from llm.adapters.claude import ClaudeAdapter
from llm.adapters.gemini import GeminiAdapter
from llm.adapters.openai import OpenAIAdapter


def get_llm_client() -> BaseLLMAdapter:
    """
    Get the configured LLM client based on MODEL_PROVIDER environment variable.
    
    Returns:
        Configured LLM adapter (Claude, OpenAI, or Gemini).
        
    Raises:
        ValueError: If MODEL_PROVIDER is invalid or API key is missing.
    """
    provider = os.getenv("MODEL_PROVIDER", "claude").lower()
    
    if provider == "claude":
        return ClaudeAdapter()
    elif provider == "openai":
        return OpenAIAdapter()
    elif provider == "gemini":
        return GeminiAdapter()
    else:
        raise ValueError(
            f"Invalid MODEL_PROVIDER: {provider}. "
            f"Must be 'claude', 'openai', or 'gemini'."
        )
