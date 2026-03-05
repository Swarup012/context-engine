"""OpenAI adapter."""

import os
from typing import Iterator

from openai import OpenAI

from llm.adapters.base import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI's GPT models."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI adapter.
        
        Args:
            model: Model name to use (default: gpt-4o-mini).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment.\n"
                "Please set it as a system environment variable:\n"
                "  export OPENAI_API_KEY='your-key-here'\n"
                "Or create a .env file in your project directory."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def complete(self, messages: list[dict], system: str = "") -> str:
        """Get a complete response from OpenAI."""
        # OpenAI uses system message in the messages list
        msgs = messages.copy()
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """Stream response chunks from OpenAI."""
        # OpenAI uses system message in the messages list
        msgs = messages.copy()
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=4096,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model
