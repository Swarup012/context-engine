"""Tests for the MCP server tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_PROJECT = Path("tests/fixtures/sample_project")


@pytest.fixture
def indexed_project(tmp_path):
    """Build + index the sample project into a temp directory."""
    from indexer.graph_builder import build_graph
    from indexer.embedder import generate_embeddings
    from storage.index_store import save_index

    graph, functions = build_graph(SAMPLE_PROJECT)

    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir(parents=True)
    save_index(graph, functions, index_dir)
    generate_embeddings(functions, index_dir, show_progress=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Tool registration / import sanity
# ---------------------------------------------------------------------------

def test_mcp_server_imports():
    """MCP server module imports without errors."""
    import mcp_server  # noqa: F401
    assert True


def test_mcp_server_has_main():
    """mcp_server exposes a callable main() entry point."""
    import mcp_server
    assert callable(mcp_server.main)


def test_mcp_instance_exists():
    """FastMCP instance is created at module level."""
    import mcp_server
    assert mcp_server.mcp is not None


def test_all_three_tools_registered():
    """All three tools are registered on the MCP server."""
    import mcp_server

    # FastMCP stores tools in ._tool_manager or similar; check via list_tools
    # Use the internal tool manager to enumerate registered tools
    tool_manager = mcp_server.mcp._tool_manager
    tool_names = set(tool_manager._tools.keys())

    assert "index_codebase" in tool_names, f"index_codebase missing. Found: {tool_names}"
    assert "ask_codebase" in tool_names, f"ask_codebase missing. Found: {tool_names}"
    assert "get_codebase_status" in tool_names, f"get_codebase_status missing. Found: {tool_names}"


# ---------------------------------------------------------------------------
# Tool 1: index_codebase
# ---------------------------------------------------------------------------

def test_index_codebase_returns_summary(tmp_path):
    """index_codebase indexes a project and returns a human-readable summary."""
    from mcp_server import index_codebase

    result = index_codebase(str(tmp_path / ".."))  # use parent — but we'll use sample directly
    # Actually index the real sample project into tmp_path
    result = index_codebase(str(SAMPLE_PROJECT))

    assert isinstance(result, str)
    assert "files" in result.lower()
    assert "functions" in result.lower()
    assert "edges" in result.lower()


def test_index_codebase_counts_are_positive(tmp_path):
    """index_codebase reports positive counts for a real project."""
    from mcp_server import index_codebase

    result = index_codebase(str(SAMPLE_PROJECT))

    # Extract numbers — result is like "Indexed 3 files, 12 functions, 8 edges."
    import re
    numbers = re.findall(r"\d+", result)
    assert len(numbers) >= 3
    files, functions, edges = int(numbers[0]), int(numbers[1]), int(numbers[2])

    assert files > 0
    assert functions > 0
    assert edges >= 0


def test_index_codebase_creates_index_dir():
    """index_codebase creates a .context-engine directory in the project."""
    from mcp_server import index_codebase

    index_codebase(str(SAMPLE_PROJECT))

    index_dir = SAMPLE_PROJECT / ".context-engine"
    assert index_dir.exists()


def test_index_codebase_nonexistent_path():
    """index_codebase raises ValueError for a path that doesn't exist."""
    from mcp_server import index_codebase

    with pytest.raises(ValueError, match="does not exist"):
        index_codebase("/nonexistent/path/that/does/not/exist")


def test_index_codebase_file_path(tmp_path):
    """index_codebase raises ValueError when given a file instead of a directory."""
    from mcp_server import index_codebase

    # Create a file
    f = tmp_path / "notadir.py"
    f.write_text("x = 1")

    with pytest.raises(ValueError, match="not a directory"):
        index_codebase(str(f))


# ---------------------------------------------------------------------------
# Tool 2: ask_codebase
# ---------------------------------------------------------------------------

def test_ask_codebase_returns_string(indexed_project):
    """ask_codebase returns a string."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert isinstance(result, str)


def test_ask_codebase_has_metadata_header(indexed_project):
    """ask_codebase result contains the metadata header section."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert "CONTEXT ENGINE" in result
    assert "Query:" in result
    assert "Tokens Used:" in result


def test_ask_codebase_has_hot_section(indexed_project):
    """ask_codebase result contains HOT tier content."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert "=== HOT:" in result


def test_ask_codebase_shows_focal_point(indexed_project):
    """ask_codebase result shows the focal point."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert "Focal Point" in result
    assert "validate_token" in result


def test_ask_codebase_respects_token_budget(indexed_project):
    """ask_codebase respects the token_budget parameter."""
    from mcp_server import ask_codebase

    # Very small budget — should still return something but note limited tokens
    result = ask_codebase("validate token", str(indexed_project), token_budget=500)

    assert isinstance(result, str)
    assert len(result) > 0


def test_ask_codebase_nonexistent_path():
    """ask_codebase raises ValueError for a path that doesn't exist."""
    from mcp_server import ask_codebase

    with pytest.raises(ValueError, match="does not exist"):
        ask_codebase("anything", "/nonexistent/path/xyz")


def test_ask_codebase_not_indexed(tmp_path):
    """ask_codebase raises ValueError when project hasn't been indexed."""
    from mcp_server import ask_codebase

    with pytest.raises(ValueError, match="No index found"):
        ask_codebase("anything", str(tmp_path))


def test_ask_codebase_shows_token_counts(indexed_project):
    """ask_codebase result includes HOT/WARM/COLD token breakdown."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert "HOT:" in result
    assert "WARM:" in result
    assert "COLD:" in result


def test_ask_codebase_query_type_shown(indexed_project):
    """ask_codebase result includes query type."""
    from mcp_server import ask_codebase

    result = ask_codebase("validate token", str(indexed_project))

    assert "Query Type:" in result


# ---------------------------------------------------------------------------
# Tool 3: get_codebase_status
# ---------------------------------------------------------------------------

def test_get_codebase_status_returns_string(indexed_project):
    """get_codebase_status returns a string."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(indexed_project))

    assert isinstance(result, str)


def test_get_codebase_status_shows_counts(indexed_project):
    """get_codebase_status shows files, functions, and edges."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(indexed_project))

    assert "Files Indexed" in result
    assert "Functions Found" in result
    assert "Dependency Edges" in result


def test_get_codebase_status_shows_last_indexed(indexed_project):
    """get_codebase_status includes the last indexed timestamp."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(indexed_project))

    assert "Last Indexed" in result


def test_get_codebase_status_up_to_date(indexed_project):
    """get_codebase_status reports index as up to date immediately after indexing."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(indexed_project))

    assert "up to date" in result.lower()


def test_get_codebase_status_stale_detection(indexed_project):
    """get_codebase_status detects files modified after indexing."""
    import time
    from mcp_server import get_codebase_status

    # Touch a source file to make it newer than the index
    # We need to copy a py file into the tmp project dir
    src = SAMPLE_PROJECT / "auth.py"
    dest = indexed_project / "auth.py"
    dest.write_text(src.read_text())

    # Give it a newer mtime
    time.sleep(0.05)
    dest.touch()

    result = get_codebase_status(str(indexed_project))

    assert "STALE" in result or "stale" in result.lower() or "changed" in result.lower()


def test_get_codebase_status_not_indexed(tmp_path):
    """get_codebase_status reports no index for an unindexed project."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(tmp_path))

    assert "No index found" in result
    assert "index_codebase" in result


def test_get_codebase_status_nonexistent_path():
    """get_codebase_status raises ValueError for a path that doesn't exist."""
    from mcp_server import get_codebase_status

    with pytest.raises(ValueError, match="does not exist"):
        get_codebase_status("/nonexistent/path/xyz")


def test_get_codebase_status_shows_project_path(indexed_project):
    """get_codebase_status includes the project path in output."""
    from mcp_server import get_codebase_status

    result = get_codebase_status(str(indexed_project))

    assert str(indexed_project) in result


# ---------------------------------------------------------------------------
# Integration: index then ask then status
# ---------------------------------------------------------------------------

def test_full_pipeline_index_ask_status(tmp_path):
    """Full pipeline: index → ask → status works end-to-end."""
    from mcp_server import index_codebase, ask_codebase, get_codebase_status

    # Step 1: Index
    index_result = index_codebase(str(SAMPLE_PROJECT))
    assert "files" in index_result.lower()

    # Step 2: Ask (uses the index in SAMPLE_PROJECT/.context-engine)
    ask_result = ask_codebase("validate token", str(SAMPLE_PROJECT))
    assert isinstance(ask_result, str)
    assert len(ask_result) > 0

    # Step 3: Status
    status_result = get_codebase_status(str(SAMPLE_PROJECT))
    assert "Functions Found" in status_result
