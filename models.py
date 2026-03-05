"""Data models for ContextEngine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class FunctionNode:
    """Represents a function in the codebase with metadata and dependencies."""
    
    name: str                    # e.g. "validate_token"
    qualified_name: str          # e.g. "auth.oauth.validate_token"
    file_path: Path
    line_start: int
    line_end: int
    source_code: str
    docstring: str | None
    calls: list[str]             # list of qualified_names this function calls
    imports: list[str]           # list of module names imported


@dataclass
class ContextChunk:
    """Represents a chunk of context with associated metadata."""
    
    node: FunctionNode
    tier: Literal["hot", "warm", "cold"]
    content: str                 # full source (hot), summary (warm), signature (cold)
    token_count: int
    relevance_score: float
    was_cached: bool = False     # whether compression was cached (for WARM tier)


@dataclass
class QueryAnalysis:
    """Analysis of a developer query to identify focal points and complexity."""
    
    query: str
    focal_points: list[str]      # list of qualified_names (1 to 3 max)
    query_type: str              # "single", "multi", "causal", "comparison"
    concepts: list[str]          # key concepts extracted from query
    is_complex: bool             # True if multiple systems involved


@dataclass
class AssembledContext:
    """Represents the final assembled context ready to send to an LLM."""
    
    chunks: list[ContextChunk]
    total_tokens: int
    budget_used_percent: float
    focal_point: str             # For backwards compatibility (first focal point)
    focal_points: list[str] = None  # All focal points (for multi-focal queries)
    query_analysis: QueryAnalysis = None  # Query analysis results
    cold_filtered_count: int = 0  # How many COLD candidates were filtered out
