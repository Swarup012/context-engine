"""Heuristic query understanding without LLM calls."""

from pathlib import Path

from models import QueryAnalysis
from retriever.semantic_search import semantic_search


def heuristic_query_analysis(
    query: str,
    project_path: Path,
    top_k: int = 10
) -> QueryAnalysis:
    """
    Analyze query using heuristics instead of LLM for common patterns.

    Detects:
    - Multi-focal queries ("auth and caching")
    - Causal queries ("why does X")
    - Enumeration queries ("list all X")
    - Comparison queries ("difference between X and Y")

    Args:
        query: User's query string.
        project_path: Path to the indexed project.
        top_k: Maximum number of focal points to return (default 10).

    Returns:
        QueryAnalysis with heuristic results.
    """
    query_lower = query.lower().strip()

    # Detect multi-focal: "auth and caching", "X and Y", "X + Y"
    multi_connectors = ["and", "&", "+", "with", "plus", " and "]
    is_multi = any(c in query_lower for c in multi_connectors)

    # Detect causal: "why", "because", "after", "cause", "fail"
    causal_words = ["why", "because", "after", "cause", "causes", "fail", "failure", "error"]
    is_causal = any(w in query_lower for w in causal_words)

    # Detect comparison: "difference", "vs", "versus", "compare"
    comparison_words = ["difference", "differences", "vs", "versus", "compare", "comparing", "versus"]
    is_comparison = any(w in query_lower for w in comparison_words)

    # Detect enumeration: "list all", "what are all", "find all", "show all"
    enumeration_words = ["list all", "what are all", "find all", "show all", "all commands", "all routes", "all models"]
    is_enumeration = any(w in query_lower for w in enumeration_words)

    # Determine query type
    if is_comparison:
        query_type = "comparison"
    elif is_causal:
        query_type = "causal"
    elif is_enumeration:
        query_type = "enumeration"
    elif is_multi:
        query_type = "multi"
    else:
        query_type = "single"

    # Determine focal point count based on query complexity
    # Fixed operator precedence: use explicit conditional instead of or/and chain
    is_complex = is_multi or is_causal or is_comparison or is_enumeration

    # Use semantic search to find focal points (no LLM needed)
    # More focal points for complex queries, less for simple queries
    search_top_k = 5 if is_complex else 1

    search_results = semantic_search(query, project_path, top_k=search_top_k)

    if not search_results:
        # No results found — empty analysis
        return QueryAnalysis(
            query=query,
            focal_points=[],
            query_type=query_type,
            concepts=[],
            is_complex=is_complex,
            focal_count=0,
        )

    # Extract focal points from search results
    focal_points = [func.qualified_name for score, func in search_results]

    # Simple keyword extraction for concepts
    # Remove common stop words and short terms
    stop_words = {
        "what", "how", "why", "where", "when", "who", "which", "the", "a", "an",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "to", "for", "of", "in", "on", "at", "by", "with", "from",
        "and", "or", "but", "so", "because", "after",
        "list", "find", "show", "all", "any", "some"
    }

    # Extract nouns/proper nouns as concepts (simple heuristic)
    words = query_lower.split()
    concepts = [w for w in words if len(w) > 3 and w not in stop_words]

    # Remove duplicates while preserving order
    concepts = list(dict.fromkeys(concepts))

    return QueryAnalysis(
        query=query,
        focal_points=focal_points,
        query_type=query_type,
        concepts=concepts,
        is_complex=is_complex,
    )