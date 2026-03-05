"""Tests for COLD tier smart filtering using semantic similarity scoring.

Tests verify:
- Irrelevant functions filtered below score threshold
- Similarity threshold (>0.3) applied correctly
- Maximum 20 COLD functions enforced
- Empty COLD tier handled gracefully
- Per-tier relevance scores correct (HOT=1.0, WARM=0.7, COLD=actual score)
"""

from pathlib import Path
from unittest import mock

import pytest

from assembler.context_builder import (
    COLD_MAX_COUNT,
    COLD_SIMILARITY_THRESHOLD,
    score_cold_candidates,
)
from indexer.embedder import generate_embeddings
from indexer.graph_builder import build_graph
from models import ContextChunk, FunctionNode
from storage.index_store import save_index


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def indexed_sample_project(tmp_path):
    """Create an indexed sample project for testing (real ChromaDB)."""
    sample_project = Path("tests/fixtures/sample_project")

    graph, functions = build_graph(sample_project)

    index_dir = tmp_path / ".context-engine"
    index_dir.mkdir(parents=True)

    save_index(graph, functions, index_dir)
    generate_embeddings(functions, index_dir, show_progress=False)

    return tmp_path, functions


def _make_function_node(name: str) -> FunctionNode:
    """Helper: build a minimal FunctionNode for testing."""
    return FunctionNode(
        name=name,
        qualified_name=f"mod.{name}",
        file_path=Path("mod.py"),
        line_start=1,
        line_end=5,
        source_code=f"def {name}():\n    pass",
        docstring=None,
        calls=[],
        imports=[],
    )


# ---------------------------------------------------------------------------
# Unit tests — mock ChromaDB to avoid slow embedding in fast tests
# ---------------------------------------------------------------------------


def test_cold_similarity_threshold_filters_low_scores(tmp_path):
    """Functions with similarity score <= 0.3 must not appear in COLD tier result."""
    # Candidates: qname_a scores 0.5 (pass), qname_b scores 0.15 (fail)
    candidates = ["mod.high_score", "mod.low_score"]

    # Fake ChromaDB: returns both IDs with distances
    # distance = 1 - similarity  →  similarity 0.5 → distance 0.5
    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "ids": [["mod.high_score", "mod.low_score"]],
        "distances": [[0.5, 0.85]],  # 1-0.5=0.5 similarity, 1-0.85=0.15 similarity
    }

    mock_client = mock.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    # Patch ChromaDB and SentenceTransformer
    fake_embedding = [0.0] * 384
    with (
        mock.patch("assembler.context_builder.chromadb.PersistentClient", return_value=mock_client),
        mock.patch(
            "assembler.context_builder.SentenceTransformer"
        ) as mock_st,
    ):
        mock_st.return_value.encode.return_value = fake_embedding

        # Create fake chroma dir so existence check passes
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        scored, filtered_count = score_cold_candidates(
            candidates, "test query", tmp_path, set(), set()
        )

    # Only high_score (0.5 > 0.3) should pass
    qualified_names = [qn for _, qn in scored]
    assert "mod.high_score" in qualified_names
    assert "mod.low_score" not in qualified_names

    # filtered_count includes the one below threshold
    assert filtered_count == 1


def test_cold_max_20_functions_enforced(tmp_path):
    """Even with 50 candidates passing the threshold, at most 20 are returned."""
    # Build 50 candidates, all scoring 0.8 (well above threshold)
    candidates = [f"mod.func_{i}" for i in range(50)]

    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 50
    mock_collection.query.return_value = {
        "ids": [candidates],
        "distances": [[0.2] * 50],  # similarity = 1 - 0.2 = 0.8 for all
    }

    mock_client = mock.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    fake_embedding = [0.0] * 384
    with (
        mock.patch("assembler.context_builder.chromadb.PersistentClient", return_value=mock_client),
        mock.patch(
            "assembler.context_builder.SentenceTransformer"
        ) as mock_st,
    ):
        mock_st.return_value.encode.return_value = fake_embedding

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        scored, filtered_count = score_cold_candidates(
            candidates, "test query", tmp_path, set(), set()
        )

    assert len(scored) == COLD_MAX_COUNT, (
        f"Expected at most {COLD_MAX_COUNT} COLD functions, got {len(scored)}"
    )


def test_empty_cold_tier_when_nothing_passes_threshold(tmp_path):
    """When all candidates score <= 0.3, COLD tier is empty — no error raised."""
    candidates = ["mod.irrelevant_a", "mod.irrelevant_b"]

    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "ids": [candidates],
        "distances": [[0.9, 0.95]],  # similarity = 0.1 and 0.05 — both below threshold
    }

    mock_client = mock.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    fake_embedding = [0.0] * 384
    with (
        mock.patch("assembler.context_builder.chromadb.PersistentClient", return_value=mock_client),
        mock.patch(
            "assembler.context_builder.SentenceTransformer"
        ) as mock_st,
    ):
        mock_st.return_value.encode.return_value = fake_embedding

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        scored, filtered_count = score_cold_candidates(
            candidates, "test query", tmp_path, set(), set()
        )

    # COLD tier is empty — that is fine
    assert scored == []
    # Both were filtered
    assert filtered_count == 2


def test_empty_candidates_list_returns_empty(tmp_path):
    """Passing an empty candidates list returns empty — no crash, no ChromaDB call."""
    with mock.patch("assembler.context_builder.chromadb.PersistentClient") as mock_pf:
        scored, filtered_count = score_cold_candidates(
            [], "test query", tmp_path, set(), set()
        )

    assert scored == []
    assert filtered_count == 0
    # ChromaDB should NOT be called with empty input
    mock_pf.assert_not_called()


def test_hot_warm_excluded_from_cold(tmp_path):
    """Candidates already in HOT or WARM sets must be excluded from COLD results."""
    candidates = ["mod.hot_func", "mod.warm_func", "mod.cold_func"]
    hot_set = {"mod.hot_func"}
    warm_set = {"mod.warm_func"}

    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        # All score 0.8 — but hot/warm should be excluded
        "ids": [candidates],
        "distances": [[0.2, 0.2, 0.2]],
    }

    mock_client = mock.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    fake_embedding = [0.0] * 384
    with (
        mock.patch("assembler.context_builder.chromadb.PersistentClient", return_value=mock_client),
        mock.patch(
            "assembler.context_builder.SentenceTransformer"
        ) as mock_st,
    ):
        mock_st.return_value.encode.return_value = fake_embedding

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        scored, _ = score_cold_candidates(
            candidates, "test query", tmp_path, hot_set, warm_set
        )

    qualified_names = [qn for _, qn in scored]
    assert "mod.hot_func" not in qualified_names
    assert "mod.warm_func" not in qualified_names
    assert "mod.cold_func" in qualified_names


def test_cold_sorted_by_score_descending(tmp_path):
    """COLD results are sorted by similarity score in descending order."""
    candidates = ["mod.a", "mod.b", "mod.c"]

    mock_collection = mock.MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "ids": [["mod.a", "mod.b", "mod.c"]],
        # Distances → similarities: a=0.4, b=0.8, c=0.6
        "distances": [[0.6, 0.2, 0.4]],
    }

    mock_client = mock.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    fake_embedding = [0.0] * 384
    with (
        mock.patch("assembler.context_builder.chromadb.PersistentClient", return_value=mock_client),
        mock.patch(
            "assembler.context_builder.SentenceTransformer"
        ) as mock_st,
    ):
        mock_st.return_value.encode.return_value = fake_embedding

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()

        scored, _ = score_cold_candidates(
            candidates, "query", tmp_path, set(), set()
        )

    # Check descending order: mod.b (0.8), mod.c (0.6), mod.a (0.4)
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True)
    assert scored[0][1] == "mod.b"  # highest score first


# ---------------------------------------------------------------------------
# Integration tests — real ChromaDB, real embeddings
# ---------------------------------------------------------------------------


def test_relevance_scores_per_tier_integration(indexed_sample_project):
    """HOT=1.0 or 0.8, WARM=0.7, COLD=actual similarity score (integration)."""
    from assembler.context_builder import assemble_context

    tmp_path, _ = indexed_sample_project

    assembled = assemble_context(
        "validate token authentication",
        tmp_path,
        token_budget=150000,
    )

    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    warm_chunks = [c for c in assembled.chunks if c.tier == "warm"]
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]

    # HOT: focal gets 1.0, others get 0.8
    for chunk in hot_chunks:
        assert chunk.relevance_score in (1.0, 0.8), (
            f"HOT chunk {chunk.node.qualified_name} has unexpected score {chunk.relevance_score}"
        )

    # WARM: all get 0.7
    for chunk in warm_chunks:
        assert chunk.relevance_score == 0.7, (
            f"WARM chunk {chunk.node.qualified_name} has unexpected score {chunk.relevance_score}"
        )

    # COLD: actual similarity score, must be > threshold
    for chunk in cold_chunks:
        assert chunk.relevance_score > COLD_SIMILARITY_THRESHOLD, (
            f"COLD chunk {chunk.node.qualified_name} score {chunk.relevance_score} "
            f"is not above threshold {COLD_SIMILARITY_THRESHOLD}"
        )
        # Score should be a valid similarity (0-1 range)
        assert 0.0 <= chunk.relevance_score <= 1.0


def test_cold_filtered_count_reported(indexed_sample_project):
    """assembled.cold_filtered_count reflects actual number of filtered functions."""
    from assembler.context_builder import assemble_context

    tmp_path, functions = indexed_sample_project

    assembled = assemble_context(
        "validate token authentication",
        tmp_path,
        token_budget=150000,
    )

    # cold_filtered_count should be a non-negative integer
    assert isinstance(assembled.cold_filtered_count, int)
    assert assembled.cold_filtered_count >= 0

    # Total cold seen = included cold + filtered
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]
    # (We can't assert exact counts without knowing scores, but sanity check)
    assert len(cold_chunks) <= COLD_MAX_COUNT


def test_cold_tier_constant_values():
    """Verify the threshold and max-count constants are correct."""
    assert COLD_SIMILARITY_THRESHOLD == 0.3
    assert COLD_MAX_COUNT == 20
