"""Anthropic Claude adapter."""

import os
from typing import Iterator

from anthropic import Anthropic

from llm.adapters.base import BaseLLMAdapter


class ClaudeAdapter(BaseLLMAdapter):
    """Adapter for Anthropic's Claude models."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude adapter.
        
        Args:
            model: Model name to use (default: claude-sonnet-4-20250514).
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment.\n"
                "Please set it as a system environment variable:\n"
                "  export ANTHROPIC_API_KEY='your-key-here'\n"
                "Or create a .env file in your project directory."
            )
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def complete(self, messages: list[dict], system: str = "") -> str:
        """Get a complete response from Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system if system else None,
            messages=messages
        )
        
        return response.content[0].text
    
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """Stream response chunks from Claude."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system if system else None,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model
