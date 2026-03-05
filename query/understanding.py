"""Query understanding and multi-focal point detection."""

import json
import logging
import re
from pathlib import Path

from llm.adapters.base import BaseLLMAdapter
from models import QueryAnalysis
from retriever.semantic_search import semantic_search

logger = logging.getLogger(__name__)


QUERY_ANALYSIS_SYSTEM_PROMPT = """You are analyzing a developer's question about a codebase.
Extract the key technical concepts and determine if the question involves multiple parts of the codebase.

Respond in JSON only:
{
  "concepts": ["concept1", "concept2"],
  "query_type": "single|multi|causal|comparison|enumeration",
  "is_complex": true|false,
  "focal_count": 1-3
}

query_type rules:
- single: question about one feature or function
- multi: question involves two or more separate features
- causal: 'why does X cause Y', 'after X happens Y breaks'
- comparison: 'difference between X and Y'
- enumeration: 'list all X', 'what are all the X', 'show all X', 'what commands/routes/models exist'

focal_count: number of different starting points needed (1-3)
is_complex: true if multiple systems/features involved"""


# Patterns that indicate enumeration queries
_ENUMERATION_PATTERNS = [
    r"\blist all\b",
    r"\bwhat are (all )?the\b",
    r"\bshow all\b",
    r"\ball (the )?(commands?|routes?|endpoints?|models?|functions?|methods?|apis?|handlers?|controllers?|services?|components?|hooks?|tests?|files?)\b",
    r"\bwhat (commands?|routes?|endpoints?|models?|functions?|methods?|apis?|handlers?|controllers?|services?|components?|hooks?) (exist|are (available|there|defined|implemented))\b",
    r"\bhow many\b",
    r"\benumerate\b",
    r"\bfind all\b",
    r"\bget all\b",
]


def _is_enumeration_query(query: str) -> bool:
    """Detect if a query is asking to list/enumerate multiple things."""
    q = query.lower()
    return any(re.search(pattern, q) for pattern in _ENUMERATION_PATTERNS)


def analyze_query(
    query: str,
    project_path: Path,
    llm_adapter: BaseLLMAdapter
) -> QueryAnalysis:
    """
    Analyze a developer query to identify focal points and complexity.
    
    Uses LLM to understand the query intent and semantic search to find
    multiple focal points if needed.
    
    Args:
        query: The developer's question.
        project_path: Path to the indexed project.
        llm_adapter: LLM adapter for query analysis.
        
    Returns:
        QueryAnalysis with focal points, query type, and concepts.
    """
    logger.info(f"Analyzing query: {query}")
    
    # Step 1: Use LLM to analyze the query
    try:
        messages = [
            {"role": "user", "content": f"Analyze this developer question:\n\n{query}"}
        ]
        
        response = llm_adapter.complete(messages, system=QUERY_ANALYSIS_SYSTEM_PROMPT)
        
        # Parse JSON response
        # Handle both plain JSON and JSON in code blocks
        response_clean = response.strip()
        if response_clean.startswith("```"):
            # Extract JSON from code block
            lines = response_clean.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_json = not in_json
                    continue
                if in_json or (not line.strip().startswith("```")):
                    json_lines.append(line)
            response_clean = "\n".join(json_lines)
        
        analysis_data = json.loads(response_clean)
        
        concepts = analysis_data.get("concepts", [])
        query_type = analysis_data.get("query_type", "single")
        is_complex = analysis_data.get("is_complex", False)
        focal_count = min(analysis_data.get("focal_count", 1), 3)  # Cap at 3
        
        logger.info(f"Query analysis: type={query_type}, complex={is_complex}, focal_count={focal_count}")
        
    except Exception as e:
        logger.warning(f"LLM query analysis failed: {e}, using fallback")
        # Fallback: treat as single focal point query
        concepts = query.split()[:3]  # Use first 3 words as concepts
        query_type = "single"
        is_complex = False
        focal_count = 1
    
    # Step 2: Find focal points using semantic search
    focal_points = []
    seen_qualified_names = set()

    # Override: enumeration queries need broad multi-focal coverage
    is_enumeration = _is_enumeration_query(query) or query_type == "enumeration"
    if is_enumeration:
        query_type = "enumeration"
        is_complex = True
        logger.info("Enumeration query detected — using broad multi-focal search")

    if is_enumeration:
        # For "list all X" queries: get top-5 semantic matches as focal points
        # so graph traversal fans out across the whole codebase
        results = semantic_search(query, project_path, top_k=5)
        for _, func in results:
            if func.qualified_name not in seen_qualified_names:
                focal_points.append(func.qualified_name)
                seen_qualified_names.add(func.qualified_name)
        # Also search with each concept to catch more entry points
        for concept in concepts[:3]:
            concept_results = semantic_search(concept, project_path, top_k=3)
            for _, func in concept_results:
                if func.qualified_name not in seen_qualified_names and len(focal_points) < 8:
                    focal_points.append(func.qualified_name)
                    seen_qualified_names.add(func.qualified_name)
        # Cap enumeration focal points at 5 (enough for broad coverage)
        focal_points = focal_points[:5]
    elif focal_count == 1:
        # Simple single focal point
        results = semantic_search(query, project_path, top_k=1)
        if results:
            _, func = results[0]
            focal_points.append(func.qualified_name)
    else:
        # Multi-focal: search with different concept combinations
        search_queries = []
        
        # Original query
        search_queries.append(query)
        
        # Individual concepts
        for concept in concepts[:focal_count]:
            search_queries.append(concept)
        
        # Search with each query
        for search_query in search_queries[:focal_count]:
            results = semantic_search(search_query, project_path, top_k=2)
            
            for _, func in results:
                if func.qualified_name not in seen_qualified_names:
                    focal_points.append(func.qualified_name)
                    seen_qualified_names.add(func.qualified_name)
                    
                    if len(focal_points) >= focal_count:
                        break
            
            if len(focal_points) >= focal_count:
                break
        
        # Ensure we have at least one focal point
        if not focal_points:
            results = semantic_search(query, project_path, top_k=1)
            if results:
                _, func = results[0]
                focal_points.append(func.qualified_name)
    
    # Cap non-enumeration focal points at 3
    if not is_enumeration:
        focal_points = focal_points[:3]
    
    logger.info(f"Found {len(focal_points)} focal points: {focal_points}")
    
    return QueryAnalysis(
        query=query,
        focal_points=focal_points,
        query_type=query_type,
        concepts=concepts,
        is_complex=is_complex
    )
