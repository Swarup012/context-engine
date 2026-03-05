"""Build context with hot/warm/cold tiers."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from assembler.compressor import compress_functions_parallel
from assembler.token_budget import count_tokens
from llm.client import get_llm_client
from models import AssembledContext, ContextChunk
from query.understanding import analyze_query
from retriever.graph_traversal import get_context_candidates, traverse_multi_focal
from retriever.semantic_search import semantic_search
from storage.index_store import load_index

logger = logging.getLogger(__name__)

COLD_SIMILARITY_THRESHOLD = 0.3
COLD_MAX_COUNT = 20


def score_cold_candidates(
    cold_candidates: list[str],
    query: str,
    index_dir: Path,
    hot_set: set[str],
    warm_set: set[str],
) -> tuple[list[tuple[float, str]], int]:
    """
    Score COLD tier candidates using ChromaDB semantic similarity.

    Filters out irrelevant functions (score <= threshold), sorts by
    relevance, and caps at COLD_MAX_COUNT.

    Args:
        cold_candidates: List of qualified names for potential COLD functions.
        query: The original user query string.
        index_dir: Path to the .context-engine/ directory containing ChromaDB.
        hot_set: Set of qualified names already in HOT tier (excluded).
        warm_set: Set of qualified names already in WARM tier (excluded).

    Returns:
        Tuple of:
        - List of (score, qualified_name) sorted descending, capped at COLD_MAX_COUNT
        - Count of candidates that were filtered out (score <= threshold)

    Raises:
        Never raises — falls back to empty list on any ChromaDB error.
    """
    if not cold_candidates:
        return [], 0

    chroma_dir = index_dir / "chroma"
    if not chroma_dir.exists():
        logger.warning("ChromaDB not found at %s — skipping COLD scoring", chroma_dir)
        return [], 0

    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name="functions")

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        query_embedding = model.encode(query, show_progress_bar=False)
        # Convert to plain Python floats for ChromaDB.
        # numpy arrays have .tolist() which gives [float, ...] — plain list() gives
        # [np.float32, ...] which ChromaDB rejects. Fall back to list() for mocks.
        embedding_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)

        # Query ChromaDB for ALL cold candidates at once
        # ChromaDB caps n_results at collection size, so we request the full set
        n_results = min(len(cold_candidates), collection.count())
        if n_results == 0:
            return [], len(cold_candidates)

        results = collection.query(
            query_embeddings=[embedding_list],
            n_results=n_results,
            where=None,
        )

        # Build a score map from ChromaDB results
        score_map: dict[str, float] = {}
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)
            for qname, dist in zip(ids, distances):
                score_map[qname] = 1.0 - dist  # cosine distance → similarity

        # Filter, deduplicate against HOT/WARM, threshold & cap
        scored: list[tuple[float, str]] = []
        for qname in cold_candidates:
            # Skip if already in a higher tier (shouldn't happen, but defensive)
            if qname in hot_set or qname in warm_set:
                continue
            score = score_map.get(qname, 0.0)
            if score > COLD_SIMILARITY_THRESHOLD:
                scored.append((score, qname))

        filtered_count = len(cold_candidates) - len(scored)

        # Sort descending by score, cap at max
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:COLD_MAX_COUNT]

        logger.info(
            "COLD scoring: %d candidates → %d relevant (filtered %d, threshold=%.1f)",
            len(cold_candidates),
            len(scored),
            filtered_count,
            COLD_SIMILARITY_THRESHOLD,
        )
        return scored, filtered_count

    except Exception as e:
        logger.warning("COLD tier scoring failed (%s) — COLD tier will be empty", e)
        return [], 0


def format_context_for_llm(assembled_context: AssembledContext) -> str:
    """
    Format assembled context into a clean string for LLM consumption.

    Args:
        assembled_context: The AssembledContext to format.

    Returns:
        Formatted string with all chunks organized by tier.
    """
    sections = []

    # Group chunks by tier
    hot_chunks = [c for c in assembled_context.chunks if c.tier == "hot"]
    warm_chunks = [c for c in assembled_context.chunks if c.tier == "warm"]
    cold_chunks = [c for c in assembled_context.chunks if c.tier == "cold"]

    # Format HOT tier
    for chunk in hot_chunks:
        func = chunk.node
        file_display = f"{func.file_path.name}"

        sections.append(f"=== HOT: {func.qualified_name} ({file_display}:{func.line_start}) ===")
        sections.append(chunk.content)
        sections.append("")  # blank line

    # Format WARM tier
    for chunk in warm_chunks:
        func = chunk.node
        file_display = f"{func.file_path.name}"

        sections.append(f"=== WARM: {func.qualified_name} ({file_display}:{func.line_start}) ===")
        sections.append(chunk.content)
        sections.append("")  # blank line

    # Format COLD tier
    for chunk in cold_chunks:
        func = chunk.node

        sections.append(f"=== COLD: {func.qualified_name} ===")
        sections.append(chunk.content)
        sections.append("")  # blank line

    return "\n".join(sections)


def assemble_context(
    query: str,
    project_path: Path,
    token_budget: int = 150000,
) -> AssembledContext:
    """
    Assemble context with hot/warm/cold tiers based on a query.

    Uses multi-focal analysis for complex queries involving multiple systems.
    COLD tier is filtered using ChromaDB semantic similarity (threshold 0.3,
    max 20 functions) to eliminate irrelevant noise.

    Args:
        query: The user's query string.
        project_path: Path to the indexed project.
        token_budget: Maximum number of tokens allowed (default 150k).

    Returns:
        AssembledContext with chunks, total_tokens, budget_used_percent,
        focal_point(s), and cold_filtered_count.
    """
    # Step 1: Load index and get graph
    index_dir = project_path / ".context-engine"
    graph, functions, _ = load_index(index_dir)

    # Step 2: Analyze query to identify focal points and complexity
    try:
        llm_adapter = get_llm_client()
        query_analysis = analyze_query(query, project_path, llm_adapter)

        logger.info(
            "Query analysis: type=%s, complex=%s, focal_points=%d",
            query_analysis.query_type,
            query_analysis.is_complex,
            len(query_analysis.focal_points),
        )
    except Exception as e:
        logger.warning("Query analysis failed: %s, using fallback single focal point", e)
        # Fallback: use simple semantic search
        search_results = semantic_search(query, project_path, top_k=1)

        if not search_results:
            return AssembledContext(
                chunks=[],
                total_tokens=0,
                budget_used_percent=0.0,
                focal_point="<none>",
            )

        focal_score, focal_func = search_results[0]
        focal_qualified_name = focal_func.qualified_name

        from models import QueryAnalysis

        query_analysis = QueryAnalysis(
            query=query,
            focal_points=[focal_qualified_name],
            query_type="single",
            concepts=[],
            is_complex=False,
        )

    # Ensure we have at least one focal point
    if not query_analysis.focal_points:
        return AssembledContext(
            chunks=[],
            total_tokens=0,
            budget_used_percent=0.0,
            focal_point="<none>",
            focal_points=[],
            query_analysis=query_analysis,
        )

    focal_qualified_name = query_analysis.focal_points[0]  # First for backwards compatibility

    # Step 3: Get hot/warm/cold candidates from graph traversal
    if query_analysis.is_complex or query_analysis.query_type in ("multi", "causal", "comparison", "enumeration"):
        logger.info("Using multi-focal traversal for %d focal points", len(query_analysis.focal_points))
        hot_candidates, warm_candidates, cold_candidates = traverse_multi_focal(
            query_analysis.focal_points, graph, depth=2
        )
    else:
        logger.info("Using single focal point: %s", focal_qualified_name)
        hot_candidates, warm_candidates, cold_candidates = get_context_candidates(
            focal_qualified_name, graph, depth=2
        )

    logger.info(
        "Candidates — Hot: %d, Warm: %d, Cold: %d (before filtering)",
        len(hot_candidates),
        len(warm_candidates),
        len(cold_candidates),
    )

    # Step 4: Build ContextChunks for each tier
    chunks: list[ContextChunk] = []

    # ------------------------------------------------------------------
    # HOT tier: full source code
    # relevance_score: 1.0 for focal node, 0.8 for 1-hop neighbors
    # ------------------------------------------------------------------
    for qualified_name in hot_candidates:
        if qualified_name not in functions:
            continue

        func = functions[qualified_name]
        content = func.source_code
        token_count = count_tokens(content)

        chunk = ContextChunk(
            node=func,
            tier="hot",
            content=content,
            token_count=token_count,
            relevance_score=1.0 if qualified_name == focal_qualified_name else 0.8,
        )
        chunks.append(chunk)

    # Calculate remaining budget after hot chunks
    hot_tokens = sum(c.token_count for c in chunks)
    warm_budget_limit = int(token_budget * 0.8) - hot_tokens  # Fill to 80%

    logger.info("Hot tier: %d chunks, %d tokens", len(chunks), hot_tokens)

    # ------------------------------------------------------------------
    # WARM tier: LLM-compressed summaries
    # relevance_score: 0.7 for all WARM functions
    # ------------------------------------------------------------------
    warm_chunks: list[ContextChunk] = []
    warm_tokens = 0

    warm_funcs = [functions[qn] for qn in warm_candidates if qn in functions]

    if warm_funcs:
        try:
            llm_adapter = get_llm_client()

            logger.info("Compressing %d WARM tier functions...", len(warm_funcs))
            compressions = compress_functions_parallel(
                warm_funcs,
                llm_adapter,
                index_dir,
                use_cache=True,
            )

            for func in warm_funcs:
                if func.qualified_name not in compressions:
                    continue

                compressed_content, was_cached = compressions[func.qualified_name]
                token_count = count_tokens(compressed_content)

                if warm_tokens + token_count <= warm_budget_limit:
                    chunk = ContextChunk(
                        node=func,
                        tier="warm",
                        content=compressed_content,
                        token_count=token_count,
                        relevance_score=0.7,
                    )
                    chunk.was_cached = was_cached
                    warm_chunks.append(chunk)
                    warm_tokens += token_count
                else:
                    break  # Warm budget exhausted

        except Exception as e:
            logger.warning("Failed to compress WARM tier: %s, falling back to truncation", e)

            # Fallback: truncation
            for qualified_name in warm_candidates:
                if qualified_name not in functions:
                    continue

                func = functions[qualified_name]

                content_parts = []
                if func.docstring:
                    content_parts.append(f'"""{func.docstring}"""')

                source_lines = func.source_code.split("\n")[:5]
                content_parts.extend(source_lines)

                content = "\n".join(content_parts)
                token_count = count_tokens(content)

                if warm_tokens + token_count <= warm_budget_limit:
                    chunk = ContextChunk(
                        node=func,
                        tier="warm",
                        content=content,
                        token_count=token_count,
                        relevance_score=0.7,
                    )
                    chunk.was_cached = False
                    warm_chunks.append(chunk)
                    warm_tokens += token_count
                else:
                    break

    chunks.extend(warm_chunks)
    logger.info("Warm tier: %d chunks, %d tokens", len(warm_chunks), warm_tokens)

    # Calculate remaining budget after warm
    remaining = token_budget - hot_tokens - warm_tokens

    # ------------------------------------------------------------------
    # COLD tier: semantically filtered signatures
    # Only functions with similarity > 0.3, sorted desc, max 20
    # relevance_score: actual similarity score from ChromaDB
    # ------------------------------------------------------------------
    hot_set = set(hot_candidates)
    warm_set = set(warm_candidates)

    scored_cold, cold_filtered_count = score_cold_candidates(
        cold_candidates, query, index_dir, hot_set, warm_set
    )

    cold_chunks: list[ContextChunk] = []
    cold_tokens = 0

    for score, qualified_name in scored_cold:
        if qualified_name not in functions:
            cold_filtered_count += 1
            continue

        func = functions[qualified_name]

        # Build cold content: function signature + first line of docstring.
        # Use the first meaningful line of source (works for Python, JS, TS, JSX, TSX).
        source_lines = func.source_code.split("\n")

        signature = ""
        for line in source_lines:
            stripped = line.strip()
            # Skip blank lines and lone braces/brackets
            if not stripped or stripped in ("{", "}", "(", ")", "[", "]"):
                continue
            signature = line
            break

        content_parts = [signature]
        if func.docstring:
            first_doc_line = func.docstring.split("\n")[0].strip()
            content_parts.append(f"    # {first_doc_line}")

        content = "\n".join(content_parts)
        token_count = count_tokens(content)

        if cold_tokens + token_count <= remaining:
            chunk = ContextChunk(
                node=func,
                tier="cold",
                content=content,
                token_count=token_count,
                relevance_score=score,
            )
            cold_chunks.append(chunk)
            cold_tokens += token_count
        else:
            break  # Budget exhausted

    chunks.extend(cold_chunks)
    logger.info(
        "Cold tier: %d chunks, %d tokens (filtered %d irrelevant)",
        len(cold_chunks),
        cold_tokens,
        cold_filtered_count,
    )

    # Step 5: Build and return AssembledContext
    total_tokens = hot_tokens + warm_tokens + cold_tokens
    budget_used_percent = (total_tokens / token_budget) * 100

    return AssembledContext(
        chunks=chunks,
        total_tokens=total_tokens,
        budget_used_percent=budget_used_percent,
        focal_point=focal_qualified_name,
        focal_points=query_analysis.focal_points,
        query_analysis=query_analysis,
        cold_filtered_count=cold_filtered_count,
    )
