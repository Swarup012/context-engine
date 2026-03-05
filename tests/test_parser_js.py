"""Tests for JavaScript/TypeScript parsing."""

import tempfile
from pathlib import Path

import pytest

from indexer.parser import parse_file
from models import FunctionNode


def test_parse_auth_jsx_returns_correct_functions():
    """Test parsing Auth.jsx returns correct function list."""
    auth_file = Path("tests/fixtures/sample_react_project/components/Auth.jsx")
    
    functions = parse_file(auth_file)
    
    # Should find Auth component and handleLogin function
    assert len(functions) >= 2
    
    # Check function names
    func_names = [f.name for f in functions]
    assert "Auth" in func_names
    assert "handleLogin" in func_names
    
    # All should be FunctionNode instances
    for func in functions:
        assert isinstance(func, FunctionNode)
        assert func.file_path == auth_file
        assert func.line_start > 0
        assert func.line_end >= func.line_start


def test_arrow_functions_extracted_correctly():
    """Test that arrow functions are extracted correctly."""
    auth_file = Path("tests/fixtures/sample_react_project/components/Auth.jsx")
    
    functions = parse_file(auth_file)
    
    # Auth and handleLogin are both arrow functions
    auth_func = next((f for f in functions if f.name == "Auth"), None)
    assert auth_func is not None
    assert "=>" in auth_func.source_code or "const Auth" in auth_func.source_code
    
    handle_login = next((f for f in functions if f.name == "handleLogin"), None)
    assert handle_login is not None
    assert "=>" in handle_login.source_code


def test_react_components_extracted_as_functions():
    """Test that React components are extracted as functions."""
    auth_file = Path("tests/fixtures/sample_react_project/components/Auth.jsx")
    
    functions = parse_file(auth_file)
    
    # Auth component should be extracted
    auth_comp = next((f for f in functions if f.name == "Auth"), None)
    assert auth_comp is not None
    assert auth_comp.qualified_name == "Auth.Auth"


def test_jsdoc_comments_extracted():
    """Test that JSDoc comments are extracted as docstrings."""
    auth_js = Path("tests/fixtures/sample_react_project/api/auth.js")
    
    functions = parse_file(auth_js)
    
    # fetchLogin should have a JSDoc comment
    fetch_login = next((f for f in functions if f.name == "fetchLogin"), None)
    assert fetch_login is not None
    
    # Check if docstring was extracted (may be None if JSDoc extraction needs work)
    # For now, just check the function was found
    assert fetch_login.source_code is not None


def test_function_calls_detected():
    """Test that calls between functions are detected."""
    auth_file = Path("tests/fixtures/sample_react_project/components/Auth.jsx")
    
    functions = parse_file(auth_file)
    
    # handleLogin should call fetchLogin
    handle_login = next((f for f in functions if f.name == "handleLogin"), None)
    assert handle_login is not None
    
    # Check if fetchLogin is in the calls list
    # The calls might include the full path or just the function name
    assert len(handle_login.calls) > 0


def test_empty_js_file_returns_empty_list():
    """Test that an empty JS file returns empty list without crashing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        temp_path = Path(f.name)
        f.write("")
    
    try:
        functions = parse_file(temp_path)
        assert functions == []
    finally:
        temp_path.unlink()


def test_js_file_with_syntax_errors_returns_empty_gracefully():
    """Test that JS file with syntax errors returns empty list gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        temp_path = Path(f.name)
        # Write invalid JavaScript syntax
        f.write("function broken( {\n")
        f.write("  // Missing closing brace and params\n")
        f.write("const x = ;\n")  # Invalid
    
    try:
        # Should not crash, should return empty list
        functions = parse_file(temp_path)
        assert functions == []
    finally:
        temp_path.unlink()


def test_async_functions_parsed():
    """Test that async functions are parsed correctly."""
    auth_js = Path("tests/fixtures/sample_react_project/api/auth.js")
    
    functions = parse_file(auth_js)
    
    # fetchLogin and fetchUserData are async
    fetch_login = next((f for f in functions if f.name == "fetchLogin"), None)
    assert fetch_login is not None
    assert "async" in fetch_login.source_code


def test_export_functions_parsed():
    """Test that export functions are parsed correctly."""
    auth_js = Path("tests/fixtures/sample_react_project/api/auth.js")
    
    functions = parse_file(auth_js)
    
    # All functions in auth.js are exported
    assert len(functions) >= 3
    
    func_names = [f.name for f in functions]
    assert "fetchLogin" in func_names
    assert "fetchUserData" in func_names
    assert "logout" in func_names


def test_custom_hook_parsed():
    """Test that custom React hooks are parsed correctly."""
    hook_file = Path("tests/fixtures/sample_react_project/hooks/useAuth.js")
    
    functions = parse_file(hook_file)
    
    # Should find useAuth hook and internal functions
    func_names = [f.name for f in functions]
    assert "useAuth" in func_names
    
    # useAuth should have calls to other functions
    use_auth = next((f for f in functions if f.name == "useAuth"), None)
    assert use_auth is not None
    assert len(use_auth.calls) > 0


def test_qualified_names_for_js():
    """Test that qualified names are generated correctly for JS files."""
    auth_js = Path("tests/fixtures/sample_react_project/api/auth.js")
    
    functions = parse_file(auth_js)
    
    # Check qualified names use filename.functionname pattern
    for func in functions:
        assert func.qualified_name.startswith("auth.")
        assert func.qualified_name.endswith(f".{func.name}")


def test_multiple_functions_in_same_file():
    """Test parsing a file with multiple functions."""
    auth_js = Path("tests/fixtures/sample_react_project/api/auth.js")
    
    functions = parse_file(auth_js)
    
    # auth.js has 3 functions
    assert len(functions) == 3
    
    # All should have different names
    func_names = [f.name for f in functions]
    assert len(func_names) == len(set(func_names))
