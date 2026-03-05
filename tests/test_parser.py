"""Tests for the parser module."""

import tempfile
from pathlib import Path

import pytest

from indexer.parser import parse_file
from models import FunctionNode


def test_parse_normal_file():
    """Test parsing a normal Python file returns correct FunctionNode list."""
    # Use the auth.py fixture
    auth_file = Path("tests/fixtures/sample_project/auth.py")
    
    functions = parse_file(auth_file)
    
    # Should find 3 functions
    assert len(functions) == 3
    
    # Check function names
    func_names = [f.name for f in functions]
    assert "validate_token" in func_names
    assert "check_token_format" in func_names
    assert "generate_token" in func_names
    
    # Check that all are FunctionNode instances
    for func in functions:
        assert isinstance(func, FunctionNode)
        assert func.file_path == auth_file
        assert func.line_start > 0
        assert func.line_end >= func.line_start
        assert len(func.source_code) > 0


def test_parse_file_with_function_calls():
    """Test that function calls are detected correctly."""
    auth_file = Path("tests/fixtures/sample_project/auth.py")
    
    functions = parse_file(auth_file)
    
    # Find validate_token function
    validate_token = next(f for f in functions if f.name == "validate_token")
    
    # It should have detected the call to check_token_format
    assert "check_token_format" in validate_token.calls


def test_parse_file_with_docstrings():
    """Test that docstrings are extracted correctly."""
    auth_file = Path("tests/fixtures/sample_project/auth.py")
    
    functions = parse_file(auth_file)
    
    # Find validate_token function
    validate_token = next(f for f in functions if f.name == "validate_token")
    
    # Should have a docstring
    assert validate_token.docstring is not None
    assert "Validate an authentication token" in validate_token.docstring


def test_parse_file_line_numbers():
    """Test that line numbers are correct."""
    auth_file = Path("tests/fixtures/sample_project/auth.py")
    
    functions = parse_file(auth_file)
    
    # All functions should have valid line numbers
    for func in functions:
        assert func.line_start > 0
        assert func.line_end > func.line_start
        
    # Line numbers should be in order (validate_token comes first in the file)
    validate_token = next(f for f in functions if f.name == "validate_token")
    check_token_format = next(f for f in functions if f.name == "check_token_format")
    
    assert validate_token.line_start < check_token_format.line_start


def test_parse_empty_file():
    """Test parsing an empty file returns empty list without crashing."""
    # Create a temporary empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_path = Path(f.name)
        # Write nothing or just whitespace
        f.write("")
    
    try:
        functions = parse_file(temp_path)
        assert functions == []
    finally:
        temp_path.unlink()


def test_parse_file_with_syntax_errors():
    """Test parsing a file with syntax errors returns empty list without crashing."""
    # Create a temporary file with syntax errors
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_path = Path(f.name)
        # Write invalid Python syntax
        f.write("def invalid_function(\n")
        f.write("    # Missing closing parenthesis and body\n")
        f.write("if True\n")  # Missing colon
        f.write("    print('broken')\n")
    
    try:
        # Should not crash, should return empty list
        functions = parse_file(temp_path)
        assert functions == []
    finally:
        temp_path.unlink()


def test_parse_nonexistent_file():
    """Test parsing a nonexistent file raises FileNotFoundError."""
    fake_path = Path("nonexistent_file.py")
    
    with pytest.raises(FileNotFoundError):
        parse_file(fake_path)


def test_qualified_names():
    """Test that qualified names are generated correctly."""
    auth_file = Path("tests/fixtures/sample_project/auth.py")
    
    functions = parse_file(auth_file)
    
    # Check qualified names contain the module name
    for func in functions:
        assert func.qualified_name.startswith("auth.")
        assert func.qualified_name.endswith(f".{func.name}")
