"""Tests for semantic search functionality."""

from pathlib import Path

import pytest

from indexer.embedder import generate_embeddings
from indexer.graph_builder import build_graph
from retriever.semantic_search import semantic_search


@pytest.fixture
def indexed_sample_project(tmp_path):
    """Create an indexed sample project for testing."""
    # Use the existing sample project
    sample_project = Path("tests/fixtures/sample_project")
    
    # Build graph
    graph, functions = build_graph(sample_project)
    
    # Create temporary index directory
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir(parents=True)
    
    # Generate embeddings
    generate_embeddings(functions, index_dir, show_progress=False)
    
    # Save functions to index_dir for semantic_search to load
    from storage.index_store import save_index
    save_index(graph, functions, index_dir)
    
    return tmp_path


def test_semantic_search_validate_authentication(indexed_sample_project):
    """Test querying 'validate authentication token' returns auth.validate_token."""
    results = semantic_search(
        "validate authentication token",
        indexed_sample_project,
        top_k=5
    )
    
    # Should return at least one result
    assert len(results) > 0
    
    # Top result should be auth.validate_token
    score, func = results[0]
    assert func.qualified_name == "auth.validate_token"
    
    # Score should be reasonable (between 0 and 1)
    assert 0 <= score <= 1


def test_semantic_search_database_connection(indexed_sample_project):
    """Test querying 'database connection' returns database.py functions."""
    results = semantic_search(
        "database connection",
        indexed_sample_project,
        top_k=5
    )
    
    # Should return results
    assert len(results) > 0
    
    # At least one result should be from database module
    database_funcs = [func for _, func in results if func.qualified_name.startswith("database.")]
    assert len(database_funcs) > 0


def test_semantic_search_user_data(indexed_sample_project):
    """Test querying 'get user data' returns database.get_user."""
    results = semantic_search(
        "get user data",
        indexed_sample_project,
        top_k=5
    )
    
    # Should return results
    assert len(results) > 0
    
    # database.get_user should be in top results
    qualified_names = [func.qualified_name for _, func in results]
    assert "database.get_user" in qualified_names


def test_semantic_search_nonsense_query(indexed_sample_project):
    """Test querying nonsense string does not crash."""
    # Should not crash, just return low-scoring results
    results = semantic_search(
        "xyzzy abracadabra quantum flux capacitor",
        indexed_sample_project,
        top_k=5
    )
    
    # Should still return some results (but with low scores)
    assert isinstance(results, list)


def test_semantic_search_no_index():
    """Test that querying without an index raises clear error."""
    fake_path = Path("nonexistent_project_12345")
    
    with pytest.raises(FileNotFoundError) as exc_info:
        semantic_search("test query", fake_path, top_k=5)
    
    # Error message should be helpful
    assert "ChromaDB collection not found" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_semantic_search_top_k_limit(indexed_sample_project):
    """Test that top_k parameter limits results correctly."""
    # Request only 2 results
    results = semantic_search(
        "function",
        indexed_sample_project,
        top_k=2
    )
    
    # Should return at most 2 results
    assert len(results) <= 2


def test_semantic_search_returns_function_nodes(indexed_sample_project):
    """Test that results contain proper FunctionNode objects."""
    results = semantic_search(
        "validate token",
        indexed_sample_project,
        top_k=5
    )
    
    assert len(results) > 0
    
    for score, func in results:
        # Check score is a float
        assert isinstance(score, float)
        
        # Check func has expected attributes
        assert hasattr(func, 'qualified_name')
        assert hasattr(func, 'file_path')
        assert hasattr(func, 'line_start')
        assert hasattr(func, 'source_code')
