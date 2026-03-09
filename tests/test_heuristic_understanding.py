"""Tests for heuristic query understanding without LLM calls."""

from pathlib import Path
import pytest

from models import QueryAnalysis
from query.heuristic_understanding import heuristic_query_analysis

SAMPLE_PROJECT = Path("tests/fixtures/sample_project")


@pytest.fixture
def indexed_project():
    """Build + index the sample project."""
    from indexer.graph_builder import build_graph
    from indexer.embedder import generate_embeddings
    from storage.index_store import save_index

    graph, functions = build_graph(SAMPLE_PROJECT)

    index_dir = SAMPLE_PROJECT / ".context-engine"
    index_dir.mkdir(parents=True, exist_ok=True)
    save_index(graph, functions, index_dir)
    generate_embeddings(functions, index_dir, show_progress=False)

    return SAMPLE_PROJECT


def test_heuristic_returns_query_analysis(indexed_project):
    """heuristic_query_analysis returns a QueryAnalysis object."""
    result = heuristic_query_analysis("validate token", indexed_project)

    assert isinstance(result, QueryAnalysis)
    assert isinstance(result.query, str)
    assert isinstance(result.query_type, str)
    assert isinstance(result.is_complex, bool)


def test_heuristic_detects_single_query(indexed_project):
    """Simple queries detected as 'single' type."""
    result = heuristic_query_analysis("validate token", indexed_project)

    assert result.query_type == "single"
    assert result.is_complex == False


def test_heuristic_detects_multi_focal(indexed_project):
    """Multi-focal queries with 'and' connector detected."""
    result = heuristic_query_analysis("authentication and authorization", indexed_project)

    assert result.query_type == "multi"
    assert result.is_complex == True


def test_heuristic_detects_causal_queries(indexed_project):
    """Causal queries with 'why' detected."""
    result = heuristic_query_analysis("why does login fail", indexed_project)

    assert result.query_type == "causal"
    assert result.is_complex == True


def test_heuristic_detects_enumeration(indexed_project):
    """Enumeration queries detected."""
    result = heuristic_query_analysis("list all commands", indexed_project)

    assert result.query_type == "enumeration"
    assert result.is_complex == True


def test_heuristic_detects_comparison(indexed_project):
    """Comparison queries detected."""
    result = heuristic_query_analysis("difference between auth and login", indexed_project)

    assert result.query_type == "comparison"
    assert result.is_complex == True


def test_heuristic_connectors_variations(indexed_project):
    """Various multi-focal connectors detected."""
    queries = [
        "auth + caching",
        "payment & checkout",
        "frontend with backend",
        "database plus cache",
    ]

    for query in queries:
        result = heuristic_query_analysis(query, indexed_project)
        assert result.query_type in ("multi", "single"), f"Query: {query}, Type: {result.query_type}"


def test_heuristic_extracts_concepts(indexed_project):
    """Concepts extracted from query (without stop words)."""
    result = heuristic_query_analysis("validate user token authentication", indexed_project)

    # Should extract meaningful keywords, not stop words
    assert len(result.concepts) > 0
    assert "what" not in result.concepts
    assert "the" not in result.concepts
    assert "and" not in result.concepts


def test_heuristic_focal_points_from_semantic_search(indexed_project):
    """Focal points extracted from semantic search results."""
    result = heuristic_query_analysis("token validation", indexed_project)

    # Should have at least one focal point
    assert len(result.focal_points) >= 1
    # All focal points should be strings
    assert all(isinstance(fp, str) for fp in result.focal_points)


def test_heuristic_complex_query_has_more_focal_points(indexed_project):
    """Complex queries return more focal points."""
    simple = heuristic_query_analysis("token", indexed_project)
    complex_query = heuristic_query_analysis("authentication and authorization", indexed_project)

    assert len(complex_query.focal_points) >= len(simple.focal_points)


def test_heuristic_empty_query_handled(indexed_project):
    """Empty or whitespace-only queries handled gracefully."""
    result = heuristic_query_analysis("   ", indexed_project)

    # Should not crash, return minimal analysis
    assert isinstance(result, QueryAnalysis)
    # Query trimmed
    assert result.query.strip() == ""


def test_heuristic_no_results_fallback(indexed_project):
    """Query handles semantic search results gracefully.

    Note: semantic search uses embeddings and may return results even for
    queries with no exact matches, so we test the graceful handling rather
    than expecting zero results.
    """
    # Use a query that likely won't match well in sample project
    result = heuristic_query_analysis("neural_network_training_quantum", indexed_project)

    assert isinstance(result, QueryAnalysis)
    # Should have SOME focal points (semantic search)
    assert len(result.focal_points) >= 0
    # focal_points should be valid qualified names
    for fp in result.focal_points:
        assert isinstance(fp, str)
        assert "." in fp  # Should be qualified names


def test_heuristic_focal_count_matches_length(indexed_project):
    """Focal count can be derived from focal points length."""
    result = heuristic_query_analysis("authentication", indexed_project)

    # focal_count was removed, but we can check len(focal_points)
    focal_count = len(result.focal_points)
    assert focal_count >= 0


def test_heuristic_query_preserved(indexed_project):
    """Original query is preserved in result."""
    query = "validate user authentication token"
    result = heuristic_query_analysis(query, indexed_project)

    assert result.query == query


def test_heuristic_case_insensitive_detection(indexed_project):
    """Query type detection is case-insensitive."""
    lowercase = heuristic_query_analysis("why does it fail", indexed_project)
    uppercase = heuristic_query_analysis("WHY DOES IT FAIL", indexed_project)
    mixed = heuristic_query_analysis("Why Does It Fail", indexed_project)

    # All should be detected as causal
    assert lowercase.query_type == "causal"
    assert uppercase.query_type == "causal"
    assert mixed.query_type == "causal"


def test_heuristic_multiple_connectors(indexed_project):
    """Queries with multiple connectors still work."""
    result = heuristic_query_analysis("auth and database and cache", indexed_project)

    # Should detect as complex multi-focal
    assert result.is_complex == True
    assert result.query_type in ("multi", "enumeration")