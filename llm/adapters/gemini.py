"""Google Gemini adapter using the new google-genai SDK."""

import os
from typing import Iterator

from google import genai
from google.genai import types

from llm.adapters.base import BaseLLMAdapter


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google's Gemini models using the new google-genai SDK."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        """
        Initialize Gemini adapter.

        Args:
            model: Model name to use (default: gemini-3-flash).

        Raises:
            ValueError: If GEMINI_API_KEY is not set in environment.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment.\n"
                "Please set it as a system environment variable:\n"
                "  export GEMINI_API_KEY='your-key-here'\n"
                "Or create a .env file in your project directory."
            )

        # New SDK: create a Client instance (not module-level configure)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def complete(self, messages: list[dict], system: str = "") -> str:
        """
        Get a complete (non-streaming) response from Gemini.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            system: Optional system instruction string.

        Returns:
            The model's response text.
        """
        contents = self._build_contents(messages)
        config = self._build_config(system)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """
        Stream response chunks from Gemini.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            system: Optional system instruction string.

        Yields:
            Text chunks as they arrive from the model.
        """
        contents = self._build_contents(messages)
        config = self._build_config(system)

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    def get_model_name(self) -> str:
        """Get the model name string."""
        return self.model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_contents(self, messages: list[dict]) -> list[types.Content]:
        """
        Convert OpenAI-style message dicts to google-genai Content objects.

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            List of types.Content objects suitable for generate_content().
        """
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            # google-genai uses "user" and "model" (not "assistant")
            genai_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=genai_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )
        return contents

    def _build_config(self, system: str) -> types.GenerateContentConfig | None:
        """
        Build GenerateContentConfig with system instruction if provided.

        Args:
            system: System instruction string (empty string = no system prompt).

        Returns:
            GenerateContentConfig if system is set, else None.
        """
        if not system:
            return None
        return types.GenerateContentConfig(
            system_instruction=system,
        )
