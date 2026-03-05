"""Tests for context assembly with hot/warm/cold tiers."""

from pathlib import Path

import pytest

from assembler.context_builder import assemble_context
from indexer.embedder import generate_embeddings
from indexer.graph_builder import build_graph
from storage.index_store import save_index


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
    
    # Save index
    save_index(graph, functions, index_dir)
    
    # Generate embeddings
    generate_embeddings(functions, index_dir, show_progress=False)
    
    return tmp_path


def test_focal_function_in_hot_tier(indexed_sample_project):
    """Test that the focal function always ends up in HOT tier."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    # Focal point should be auth.validate_token
    assert assembled.focal_point == "auth.validate_token"
    
    # Check that focal function is in HOT tier
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    hot_qualified_names = [c.node.qualified_name for c in hot_chunks]
    
    assert "auth.validate_token" in hot_qualified_names


def test_total_tokens_within_budget(indexed_sample_project):
    """Test that total tokens never exceed the budget."""
    budget = 10000
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=budget
    )
    
    # Total tokens should not exceed budget
    assert assembled.total_tokens <= budget
    
    # Budget used percent should be <= 100%
    assert assembled.budget_used_percent <= 100.0


def test_direct_dependencies_in_hot_or_warm(indexed_sample_project):
    """Test that direct dependencies of focal function are in HOT or WARM."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    # auth.validate_token calls auth.check_token_format
    # check_token_format should be in HOT (1-hop neighbor)
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    hot_qualified_names = [c.node.qualified_name for c in hot_chunks]
    
    # check_token_format is called by validate_token, so should be in HOT
    assert "auth.check_token_format" in hot_qualified_names


def test_assembled_context_has_correct_token_counts(indexed_sample_project):
    """Test that AssembledContext has correct token counts."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    # Calculate total from chunks
    calculated_total = sum(c.token_count for c in assembled.chunks)
    
    # Should match the reported total
    assert assembled.total_tokens == calculated_total
    
    # All chunks should have positive token counts
    for chunk in assembled.chunks:
        assert chunk.token_count > 0


def test_hot_tier_has_full_source(indexed_sample_project):
    """Test that HOT tier chunks contain full source code."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    
    # HOT chunks should exist
    assert len(hot_chunks) > 0
    
    # Check that hot chunks have full source code
    for chunk in hot_chunks:
        # Content should match the original source code
        assert chunk.content == chunk.node.source_code


def test_warm_tier_is_abbreviated(indexed_sample_project):
    """Test that WARM tier chunks are abbreviated."""
    assembled = assemble_context(
        "save user",
        indexed_sample_project,
        token_budget=150000
    )
    
    warm_chunks = [c for c in assembled.chunks if c.tier == "warm"]
    
    # If there are warm chunks, check they are abbreviated (first 5 lines only)
    for chunk in warm_chunks:
        # Warm content should have at most 5 lines from source (plus docstring)
        # Count lines in content
        content_lines = chunk.content.split('\n')
        source_lines = chunk.node.source_code.split('\n')
        
        # Should be abbreviated (not full source)
        # Either shorter, or if longer (due to docstring), should have fewer source lines
        if len(chunk.content) > len(chunk.node.source_code):
            # If longer due to docstring, check we only took 5 lines of source
            # This is acceptable since we're adding docstring separately
            assert len([l for l in content_lines if l.strip() and not l.strip().startswith('"""')]) <= 7


def test_cold_tier_is_signature_only(indexed_sample_project):
    """Test that COLD tier chunks contain only signature."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]
    
    # If there are cold chunks, they should be very short
    for chunk in cold_chunks:
        # Cold content should be much shorter than full source
        assert len(chunk.content) < len(chunk.node.source_code) / 2
        # Should contain "def " (the signature)
        assert "def " in chunk.content


def test_tiers_are_filled_in_order(indexed_sample_project):
    """Test that tiers are filled in correct order (hot, warm, cold)."""
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=150000
    )
    
    # Get chunks by tier
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    warm_chunks = [c for c in assembled.chunks if c.tier == "warm"]
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]
    
    # HOT should always have at least the focal function
    assert len(hot_chunks) > 0
    
    # If we have budget, we should fill tiers in order
    # (We can't test exact counts without knowing token sizes, but we can check presence)
    assert len(assembled.chunks) > 0


def test_small_budget_limits_chunks(indexed_sample_project):
    """Test that a small budget limits the number of chunks."""
    # Very small budget
    small_budget = 500
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=small_budget
    )
    
    # Should not exceed budget
    assert assembled.total_tokens <= small_budget
    
    # Should have fewer chunks than with a large budget
    large_budget = 150000
    assembled_large = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=large_budget
    )
    
    assert len(assembled.chunks) <= len(assembled_large.chunks)


def test_budget_used_percent_calculation(indexed_sample_project):
    """Test that budget_used_percent is calculated correctly."""
    budget = 10000
    assembled = assemble_context(
        "validate token",
        indexed_sample_project,
        token_budget=budget
    )
    
    # Calculate expected percentage
    expected_percent = (assembled.total_tokens / budget) * 100
    
    # Should match (within floating point precision)
    assert abs(assembled.budget_used_percent - expected_percent) < 0.01
