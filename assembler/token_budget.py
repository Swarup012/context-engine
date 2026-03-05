"""Token counting and budget management using tiktoken."""

import tiktoken

from models import ContextChunk

# Use cl100k_base encoding (used by GPT-4, Claude uses similar token counts)
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for.
        
    Returns:
        Number of tokens.
    """
    return len(ENCODING.encode(text))


def fits_in_budget(chunks: list[ContextChunk], budget: int) -> bool:
    """
    Check if a list of chunks fits within the token budget.
    
    Args:
        chunks: List of ContextChunk objects.
        budget: Maximum number of tokens allowed.
        
    Returns:
        True if total tokens <= budget, False otherwise.
    """
    total_tokens = sum(chunk.token_count for chunk in chunks)
    return total_tokens <= budget


def get_remaining_budget(chunks: list[ContextChunk], budget: int) -> int:
    """
    Get the remaining token budget after accounting for chunks.
    
    Args:
        chunks: List of ContextChunk objects already allocated.
        budget: Maximum number of tokens allowed.
        
    Returns:
        Number of tokens remaining in budget (can be negative if over budget).
    """
    total_tokens = sum(chunk.token_count for chunk in chunks)
    return budget - total_tokens
