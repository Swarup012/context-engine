"""Tests for the graph_builder module."""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from indexer.graph_builder import build_graph
from models import FunctionNode


def test_graph_has_correct_number_of_nodes():
    """Test that the graph contains all functions from the sample project."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, functions_dict = build_graph(sample_project)
    
    # auth.py has 3 functions, database.py has 2, api.py has 2 = 7 total
    assert graph.number_of_nodes() == 7
    assert len(functions_dict) == 7
    
    # Verify all expected functions are present
    expected_functions = [
        "auth.validate_token",
        "auth.check_token_format",
        "auth.generate_token",
        "database.get_user",
        "database.save_user",
        "api.login",
        "api.register"
    ]
    
    for func_name in expected_functions:
        assert func_name in functions_dict
        assert graph.has_node(func_name)


def test_edges_between_calling_functions():
    """Test that edges exist between functions that call each other."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, functions_dict = build_graph(sample_project)
    
    # In auth.py: validate_token calls check_token_format
    assert graph.has_edge("auth.validate_token", "auth.check_token_format")


def test_api_links_to_auth_and_database():
    """Test that api.py functions correctly link to auth.py and database.py."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, functions_dict = build_graph(sample_project)
    
    # api.login calls validate_token and get_user
    assert graph.has_edge("api.login", "auth.validate_token")
    assert graph.has_edge("api.login", "database.get_user")
    
    # api.register calls save_user and generate_token
    assert graph.has_edge("api.register", "database.save_user")
    assert graph.has_edge("api.register", "auth.generate_token")


def test_node_metadata_contains_function():
    """Test that each node stores the full FunctionNode as metadata."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, functions_dict = build_graph(sample_project)
    
    # Check that nodes have function metadata
    for node in graph.nodes():
        assert "function" in graph.nodes[node]
        func = graph.nodes[node]["function"]
        assert isinstance(func, FunctionNode)
        assert func.qualified_name == node


def test_functions_dict_matches_graph_nodes():
    """Test that the functions dictionary matches the graph nodes."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, functions_dict = build_graph(sample_project)
    
    # All nodes should be in the dictionary
    for node in graph.nodes():
        assert node in functions_dict
    
    # All dictionary entries should be in the graph
    for qualified_name in functions_dict.keys():
        assert graph.has_node(qualified_name)


def test_empty_directory():
    """Test building a graph from an empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        graph, functions_dict = build_graph(temp_path)
        
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
        assert len(functions_dict) == 0


def test_directory_with_syntax_errors():
    """Test that files with syntax errors are skipped without crashing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a file with syntax errors
        bad_file = temp_path / "bad.py"
        bad_file.write_text("def broken(\n  # Missing closing paren\n")
        
        # Create a valid file
        good_file = temp_path / "good.py"
        good_file.write_text("def valid_func():\n    return True\n")
        
        # Should not crash, should skip bad file and process good file
        graph, functions_dict = build_graph(temp_path)
        
        # Should have the one function from good.py
        assert graph.number_of_nodes() == 1
        assert "good.valid_func" in functions_dict


def test_circular_dependencies_do_not_crash():
    """Test that circular dependencies are handled without infinite loops."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create files with circular dependencies
        file_a = temp_path / "module_a.py"
        file_a.write_text("""
def func_a():
    '''Function A calls function B'''
    func_b()
    return True
""")
        
        file_b = temp_path / "module_b.py"
        file_b.write_text("""
def func_b():
    '''Function B calls function A'''
    func_a()
    return True
""")
        
        # Should handle circular dependencies without crashing
        graph, functions_dict = build_graph(temp_path)
        
        # Should have both functions
        assert graph.number_of_nodes() == 2
        assert "module_a.func_a" in functions_dict
        assert "module_b.func_b" in functions_dict
        
        # Should have edges in both directions (if resolved correctly)
        # Note: Resolution depends on how well _resolve_call works
        # At minimum, it should not crash


def test_nonexistent_directory():
    """Test that nonexistent directory raises appropriate error."""
    fake_path = Path("nonexistent_directory_12345")
    
    with pytest.raises(FileNotFoundError):
        build_graph(fake_path)


def test_graph_is_directed():
    """Test that the graph is a directed graph."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, _ = build_graph(sample_project)
    
    assert isinstance(graph, nx.DiGraph)
    assert graph.is_directed()


def test_edge_count():
    """Test that the graph has the expected number of edges."""
    sample_project = Path("tests/fixtures/sample_project")
    
    graph, _ = build_graph(sample_project)
    
    # Expected edges:
    # 1. auth.validate_token -> auth.check_token_format
    # 2. api.login -> auth.validate_token
    # 3. api.login -> database.get_user
    # 4. api.register -> database.save_user
    # 5. api.register -> auth.generate_token
    # Total: 5 edges
    assert graph.number_of_edges() == 5
