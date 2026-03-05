"""Tests for LLM-based function compression."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assembler.compressor import (
    compress_function,
    compress_functions_parallel,
    invalidate_cache_for_file,
    load_compression_cache,
    save_compression_cache,
)
from models import FunctionNode


@pytest.fixture
def mock_llm_adapter():
    """Create a mock LLM adapter."""
    adapter = MagicMock()
    adapter.complete.return_value = "This function validates user credentials. It takes email and password as parameters and returns a boolean indicating success. It checks against the database."
    return adapter


@pytest.fixture
def sample_function():
    """Create a sample function node for testing."""
    return FunctionNode(
        name="validate_user",
        qualified_name="auth.validate_user",
        file_path=Path("auth.py"),
        line_start=10,
        line_end=20,
        source_code="""def validate_user(email: str, password: str) -> bool:
    '''Validate user credentials'''
    user = db.get_user(email)
    if not user:
        return False
    return user.check_password(password)""",
        docstring="Validate user credentials",
        calls=["db.get_user", "user.check_password"],
        imports=["db"]
    )


def test_compression_returns_shorter_string(mock_llm_adapter, sample_function, tmp_path):
    """Test that compression returns a string shorter than original."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    compressed, was_cached = compress_function(
        sample_function,
        mock_llm_adapter,
        index_dir,
        use_cache=False
    )
    
    # Compressed should be shorter than original
    assert len(compressed) < len(sample_function.source_code)
    
    # Should not be cached on first call
    assert was_cached is False
    
    # LLM should have been called
    assert mock_llm_adapter.complete.called


def test_caching_works(mock_llm_adapter, sample_function, tmp_path):
    """Test that caching works - second call returns cached result."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # First call - should compress
    compressed1, was_cached1 = compress_function(
        sample_function,
        mock_llm_adapter,
        index_dir,
        use_cache=True
    )
    
    assert was_cached1 is False
    assert mock_llm_adapter.complete.call_count == 1
    
    # Second call with same function - should use cache
    compressed2, was_cached2 = compress_function(
        sample_function,
        mock_llm_adapter,
        index_dir,
        use_cache=True
    )
    
    assert was_cached2 is True
    assert compressed1 == compressed2
    # LLM should not be called again
    assert mock_llm_adapter.complete.call_count == 1


def test_cache_invalidated_when_source_changes(mock_llm_adapter, sample_function, tmp_path):
    """Test that cache is invalidated when function source changes."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # Compress original function
    compressed1, _ = compress_function(
        sample_function,
        mock_llm_adapter,
        index_dir,
        use_cache=True
    )
    
    # Modify the function source
    modified_function = FunctionNode(
        name=sample_function.name,
        qualified_name=sample_function.qualified_name,
        file_path=sample_function.file_path,
        line_start=sample_function.line_start,
        line_end=sample_function.line_end,
        source_code=sample_function.source_code + "\n    # Added comment",
        docstring=sample_function.docstring,
        calls=sample_function.calls,
        imports=sample_function.imports
    )
    
    # Compress modified function - should not use cache (different source)
    compressed2, was_cached = compress_function(
        modified_function,
        mock_llm_adapter,
        index_dir,
        use_cache=True
    )
    
    assert was_cached is False
    # LLM should be called again
    assert mock_llm_adapter.complete.call_count == 2


def test_fallback_on_llm_failure(sample_function, tmp_path):
    """Test that fallback works when LLM call fails."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # Create adapter that raises an exception
    failing_adapter = MagicMock()
    failing_adapter.complete.side_effect = Exception("LLM service unavailable")
    
    # Should not crash, should return fallback
    compressed, was_cached = compress_function(
        sample_function,
        failing_adapter,
        index_dir,
        use_cache=False
    )
    
    # Should have returned something
    assert compressed is not None
    assert len(compressed) > 0
    
    # Should contain docstring or first few lines
    assert "validate" in compressed.lower() or "def " in compressed


def test_parallel_compression(mock_llm_adapter, tmp_path):
    """Test that parallel compression works for multiple functions."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # Create multiple functions
    functions = [
        FunctionNode(
            name=f"func_{i}",
            qualified_name=f"module.func_{i}",
            file_path=Path("module.py"),
            line_start=i*10,
            line_end=i*10+5,
            source_code=f"def func_{i}():\n    return {i}",
            docstring=f"Function {i}",
            calls=[],
            imports=[]
        )
        for i in range(3)
    ]
    
    # Compress in parallel
    results = compress_functions_parallel(
        functions,
        mock_llm_adapter,
        index_dir,
        use_cache=False
    )
    
    # Should have compressed all functions
    assert len(results) == 3
    
    for func in functions:
        assert func.qualified_name in results
        compressed, was_cached = results[func.qualified_name]
        assert compressed is not None


def test_cache_persistence(mock_llm_adapter, sample_function, tmp_path):
    """Test that cache persists to disk."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # Compress and cache
    compress_function(sample_function, mock_llm_adapter, index_dir, use_cache=True)
    
    # Load cache manually
    cache = load_compression_cache(index_dir)
    
    # Cache should have one entry
    assert len(cache) > 0
    
    # Save and reload
    save_compression_cache(cache, index_dir)
    reloaded_cache = load_compression_cache(index_dir)
    
    assert cache == reloaded_cache


def test_invalidate_cache_for_file(mock_llm_adapter, tmp_path):
    """Test that cache invalidation removes entries for specific file."""
    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir()
    
    # Create functions from different files
    func1 = FunctionNode(
        name="func1",
        qualified_name="file1.func1",
        file_path=Path("file1.py"),
        line_start=1,
        line_end=5,
        source_code="def func1(): pass",
        docstring=None,
        calls=[],
        imports=[]
    )
    
    func2 = FunctionNode(
        name="func2",
        qualified_name="file2.func2",
        file_path=Path("file2.py"),
        line_start=1,
        line_end=5,
        source_code="def func2(): pass",
        docstring=None,
        calls=[],
        imports=[]
    )
    
    # Compress both
    compress_function(func1, mock_llm_adapter, index_dir, use_cache=True)
    compress_function(func2, mock_llm_adapter, index_dir, use_cache=True)
    
    # Cache should have 2 entries
    cache = load_compression_cache(index_dir)
    assert len(cache) == 2
    
    # Invalidate cache for file1.py
    functions = {
        func1.qualified_name: func1,
        func2.qualified_name: func2
    }
    invalidate_cache_for_file(Path("file1.py"), functions, index_dir)
    
    # Cache should now have 1 entry (func2)
    cache = load_compression_cache(index_dir)
    assert len(cache) == 1
