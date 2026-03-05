"""Abstract base class for LLM adapters."""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(self, messages: list[dict], system: str = "") -> str:
        """
        Get a complete response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt/instructions.
            
        Returns:
            The complete response text.
        """
        pass
    
    @abstractmethod
    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """
        Stream response chunks from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt/instructions.
            
        Yields:
            Response text chunks.
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used.
        
        Returns:
            Model name string.
        """
        pass
