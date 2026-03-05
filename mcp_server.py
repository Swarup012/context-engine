"""MCP Server for ContextEngine.

Exposes ContextEngine as an MCP (Model Context Protocol) server so Claude Desktop,
Cursor, and any MCP-compatible tool can use it natively.

Tools exposed:
- index_codebase: Index a project directory
- ask_codebase: Assemble intelligent context for a query
- get_codebase_status: Get index status for a project
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables (.env for local dev, system env for global installs)
if Path(".env").exists():
    load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# The working directory where Claude Code was launched — used as the default project path
_CLIENT_CWD = os.environ.get("PWD") or os.getcwd()

# Create the MCP server
mcp = FastMCP(
    "ContextEngine",
    instructions=(
        "ContextEngine gives you deep, accurate access to any indexed codebase. "
        "ALWAYS use these tools when answering questions about code, architecture, "
        "functions, bugs, or how something works in a project. "
        "Do NOT answer code questions from memory — always call ask_codebase first. "
        "Workflow: "
        "(1) If not indexed yet, call index_codebase first. "
        "(2) For ANY question about the codebase, call ask_codebase with the user's question. "
        "(3) Use get_codebase_status to check if the index is up to date. "
        f"Current project directory: {_CLIENT_CWD}. "
        "Use this path for all tool calls unless the user specifies a different path."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: index_codebase
# ---------------------------------------------------------------------------

@mcp.tool()
def index_codebase(path: str = "") -> str:
    """Index a codebase to enable intelligent context retrieval.

    Call this tool when the user says things like:
    "index this project", "index this codebase", "set up context engine",
    "analyze this codebase", or before answering questions about a project
    that hasn't been indexed yet.

    Crawls the directory, parses all Python/JS/TS files, builds a dependency
    graph, and generates semantic embeddings. The index is saved to a
    .context-engine/ folder inside the project directory.

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        path: Absolute path to the project directory to index. Leave empty to use current project.

    Returns:
        Summary string with counts of files, functions, and edges indexed,
        plus the absolute path where the index was saved.
    """
    from indexer.graph_builder import build_graph
    from indexer.embedder import generate_embeddings
    from storage.index_store import save_index

    # If no path given (or model passed garbage), fall back to the client's CWD
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved)

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not project_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    project_path = project_path.resolve()
    index_dir = project_path / ".context-engine"

    # Build dependency graph (crawl + parse)
    graph, functions = build_graph(project_path)

    # Save graph + metadata to disk
    save_index(graph, functions, index_dir)

    # Generate and store semantic embeddings
    generate_embeddings(functions, index_dir, show_progress=False)

    # Compute summary stats
    files_indexed = len(set(str(func.file_path) for func in functions.values()))
    total_functions = len(functions)
    total_edges = graph.number_of_edges()

    return (
        f"Indexed {files_indexed} files, {total_functions} functions, {total_edges} edges. "
        f"Index saved to {index_dir}"
    )


# ---------------------------------------------------------------------------
# Tool 2: ask_codebase
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_codebase(query: str, path: str = "", token_budget: int = 150000) -> str:
    """Ask a question about an indexed codebase and get assembled context.

    Call this tool for ANY question about code in the current project, including:
    "how does X work?", "what does Y function do?", "where is Z handled?",
    "explain the architecture", "list all CLI commands", "how is auth implemented?",
    "what are the API routes?", "find where X is called", "explain this bug".

    ALWAYS call this before answering code questions — do not answer from memory.

    Runs the full ContextEngine pipeline: query analysis → multi-focal graph
    traversal → HOT/WARM/COLD tier assembly → formatted context string.

    Does NOT call an LLM itself — returns assembled context so you can reason
    over it directly with full visibility into the actual source code.

    The returned context is organized in three tiers:
    - HOT: Full source code of the most relevant functions
    - WARM: Summaries of related functions
    - COLD: Signatures of peripherally relevant functions

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        query: The developer's question about the codebase.
        path: Absolute path to the indexed project directory. Leave empty to use current project.
        token_budget: Maximum tokens for assembled context (default 150000).

    Returns:
        Formatted context string with HOT/WARM/COLD sections plus metadata
        about focal points found, tokens used, and query type.
    """
    from assembler.context_builder import assemble_context, format_context_for_llm

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_codebase('{path}') first."
        )

    # Run full assembly pipeline
    assembled = assemble_context(query, project_path, token_budget=token_budget)

    if not assembled.chunks:
        return (
            f"No relevant context found for query: {query}\n\n"
            "The index may be empty or the query doesn't match any indexed functions. "
            "Try re-indexing with index_codebase() or rephrasing the query."
        )

    # Format context into sections
    formatted = format_context_for_llm(assembled)

    # Build metadata header
    meta_lines = [
        "=== CONTEXT ENGINE — ASSEMBLED CONTEXT ===",
        f"Query: {query}",
        f"Query Type: {assembled.query_analysis.query_type if assembled.query_analysis else 'single'}",
    ]

    # Focal points
    focal_points = assembled.focal_points or [assembled.focal_point]
    if len(focal_points) > 1:
        meta_lines.append(f"Focal Points ({len(focal_points)}): {', '.join(focal_points)}")
    else:
        meta_lines.append(f"Focal Point: {assembled.focal_point}")

    # Concepts
    if assembled.query_analysis and assembled.query_analysis.concepts:
        meta_lines.append(f"Concepts: {', '.join(assembled.query_analysis.concepts)}")

    # Token stats
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    warm_chunks = [c for c in assembled.chunks if c.tier == "warm"]
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]
    hot_tokens = sum(c.token_count for c in hot_chunks)
    warm_tokens = sum(c.token_count for c in warm_chunks)
    cold_tokens = sum(c.token_count for c in cold_chunks)

    meta_lines.extend([
        f"Tokens Used: {assembled.total_tokens:,} / {token_budget:,} ({assembled.budget_used_percent:.1f}%)",
        f"HOT: {len(hot_chunks)} functions ({hot_tokens:,} tokens)",
        f"WARM: {len(warm_chunks)} functions ({warm_tokens:,} tokens)",
        f"COLD: {len(cold_chunks)} functions ({cold_tokens:,} tokens)",
    ])

    if assembled.cold_filtered_count > 0:
        meta_lines.append(
            f"COLD Filtered: {assembled.cold_filtered_count} irrelevant functions removed (threshold ≥0.3)"
        )

    meta_lines.append("=" * 50)
    meta_lines.append("")

    return "\n".join(meta_lines) + formatted


# ---------------------------------------------------------------------------
# Tool 3: get_codebase_status
# ---------------------------------------------------------------------------

@mcp.tool()
def get_codebase_status(path: str = "") -> str:
    """Get the current index status of a codebase.

    Call this when the user asks things like:
    "is the index up to date?", "when was this last indexed?",
    "how many files are indexed?", "is the codebase indexed?",
    "check the index status", "has the index changed?".

    Reports how many files, functions, and edges are indexed, when the index
    was last updated, and whether any source files have changed since then
    (i.e., whether the index is stale and needs re-indexing).

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        path: Absolute path to the project directory. Leave empty to use current project.

    Returns:
        Formatted status string with index statistics and staleness info.
    """
    from storage.index_store import load_index

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        return (
            f"No index found for: {path}\n"
            "Run index_codebase() to create an index."
        )

    try:
        graph, functions, metadata = load_index(index_dir)
    except FileNotFoundError as e:
        return f"Index is incomplete or corrupt: {e}\nRun index_codebase() to rebuild."

    # Parse last indexed time
    last_indexed_str = metadata.get("last_indexed", "Unknown")
    last_indexed_display = last_indexed_str
    last_indexed_dt = None

    if last_indexed_str and last_indexed_str != "Unknown":
        try:
            last_indexed_dt = datetime.fromisoformat(last_indexed_str)
            last_indexed_display = last_indexed_dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # Check for stale files
    stale_files: list[str] = []
    if last_indexed_dt is not None:
        supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx"}
        skip_dirs = {".context-engine", "node_modules", "__pycache__", ".git", ".venv",
                     ".next", "dist", "build", "out", "coverage"}
        for file_path in project_path.rglob("*"):
            if any(part in file_path.parts for part in skip_dirs):
                continue
            if any(part.startswith(".") for part in file_path.parts):
                continue
            if file_path.is_file() and file_path.suffix in supported_exts:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > last_indexed_dt:
                    stale_files.append(str(file_path.relative_to(project_path)))

    # Build status report
    lines = [
        f"=== CODEBASE INDEX STATUS ===",
        f"Project: {project_path}",
        f"Index: {index_dir}",
        "",
        f"Files Indexed:     {metadata.get('total_files', len(set(str(f.file_path) for f in functions.values())))}",
        f"Functions Found:   {metadata.get('total_functions', len(functions))}",
        f"Dependency Edges:  {metadata.get('total_edges', graph.number_of_edges())}",
        f"Last Indexed:      {last_indexed_display}",
        "",
    ]

    if stale_files:
        lines.append(f"⚠ Index is STALE — {len(stale_files)} file(s) changed since last index:")
        for sf in stale_files[:10]:  # Show at most 10
            lines.append(f"  • {sf}")
        if len(stale_files) > 10:
            lines.append(f"  ... and {len(stale_files) - 10} more")
        lines.append("")
        lines.append("Run index_codebase() to refresh the index.")
    else:
        lines.append("✓ Index is up to date.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: search_codebase
# ---------------------------------------------------------------------------

@mcp.tool()
def search_codebase(query: str, path: str = "", top_k: int = 10) -> str:
    """Search the codebase for functions, commands, routes, or any concept.

    Call this tool when the user wants to FIND or LIST specific things, including:
    "find all CLI commands", "list all API routes", "where is X defined?",
    "find functions that handle auth", "show all models", "what files handle payments?",
    "find where X is called", "list all hooks", "show all exports".

    Unlike ask_codebase which assembles full context, this returns a quick ranked
    list of matching functions with their file locations and signatures — perfect
    for navigation and discovery.

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        query: What to search for — a concept, function name, command, route, etc.
        path: Absolute path to the indexed project directory. Leave empty to use current project.
        top_k: Number of results to return (default 10, max 20).

    Returns:
        Ranked list of matching functions with file paths, line numbers, and signatures.
    """
    from retriever.semantic_search import semantic_search

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_codebase first."
        )

    # Cap top_k at 20
    top_k = min(max(1, top_k), 20)

    results = semantic_search(query, project_path, top_k=top_k)

    if not results:
        return f"No results found for: {query}\n\nTry re-indexing with index_codebase() or rephrasing the query."

    lines = [
        f"=== SEARCH RESULTS: '{query}' ===",
        f"Found {len(results)} matches (ranked by relevance):",
        "",
    ]

    for rank, (score, func) in enumerate(results, 1):
        # File path relative to project root
        try:
            rel_path = Path(func.file_path).relative_to(project_path)
        except ValueError:
            rel_path = Path(func.file_path)

        # Build signature line
        sig = func.signature or f"function {func.name}(...)"

        lines.append(f"{rank}. {func.qualified_name}")
        lines.append(f"   File: {rel_path}:{func.start_line}")
        lines.append(f"   Signature: {sig}")
        lines.append(f"   Relevance: {score:.0%}")
        if func.docstring:
            # First line of docstring only
            doc_first = func.docstring.strip().split("\n")[0][:120]
            lines.append(f"   Doc: {doc_first}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: get_function_source
# ---------------------------------------------------------------------------

@mcp.tool()
def get_function_source(function_name: str, path: str = "") -> str:
    """Get the full source code of a specific function by name.

    Call this when the user wants to see the exact source code of a function, including:
    "show me the source of X", "what does function X look like?",
    "show the implementation of X", "display the code for X",
    "what's inside the X function?", "open X function".

    Searches by function name (partial match supported) across the indexed codebase.
    If multiple functions match, returns all of them.

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        function_name: Name of the function to look up. Can be partial (e.g. "compress"
                      matches "compressCommand") or fully qualified (e.g. "compress.compressCommand").
        path: Absolute path to the indexed project directory. Leave empty to use current project.

    Returns:
        Full source code of the matching function(s) with file path and line numbers.
    """
    from storage.index_store import load_index

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_codebase first."
        )

    _, functions, _ = load_index(index_dir)

    # Search by name — exact qualified name first, then partial match
    search_term = function_name.strip().lower()
    matches: list = []

    # Exact qualified name match
    if function_name in functions:
        matches.append(functions[function_name])
    else:
        # Partial match on qualified name or just the function name
        for qname, func in functions.items():
            if (search_term in qname.lower() or
                search_term in func.name.lower()):
                matches.append(func)

    if not matches:
        # Suggest similar names
        all_names = sorted(functions.keys())
        suggestions = [n for n in all_names if search_term[:3] in n.lower()][:5]
        suggestion_str = "\n".join(f"  • {n}" for n in suggestions) if suggestions else "  (none found)"
        return (
            f"No function named '{function_name}' found in the index.\n\n"
            f"Similar names:\n{suggestion_str}\n\n"
            f"Use search_codebase('{function_name}') to find it by concept."
        )

    lines = [
        f"=== FUNCTION SOURCE: '{function_name}' ===",
        f"Found {len(matches)} match(es):",
        "",
    ]

    for func in matches:
        try:
            rel_path = Path(func.file_path).relative_to(project_path)
        except ValueError:
            rel_path = Path(func.file_path)

        lines.append(f"{'─' * 60}")
        lines.append(f"Function: {func.qualified_name}")
        lines.append(f"File:     {rel_path}")
        lines.append(f"Lines:    {func.line_start}–{func.line_end}")
        if func.docstring:
            lines.append(f"Doc:      {func.docstring.strip().split(chr(10))[0][:120]}")
        lines.append("")
        lines.append(func.source_code)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: explain_file
# ---------------------------------------------------------------------------

@mcp.tool()
def explain_file(file_path: str, path: str = "") -> str:
    """Get a structural overview of a specific file in the codebase.

    Call this when the user asks about a specific file, including:
    "what does X.ts do?", "explain the auth module", "give me an overview of X file",
    "what's in compress.ts?", "summarize the database module",
    "what functions are in X?", "explain this file".

    Returns all functions defined in the file with their signatures, docstrings,
    what they call, and what calls them — giving a complete structural picture.
    No LLM call needed — assembles directly from the index.

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        file_path: Relative path to the file (e.g. "src/commands/compress.ts" or just "compress.ts").
        path: Absolute path to the indexed project directory. Leave empty to use current project.

    Returns:
        Structural overview of the file: purpose, functions, relationships.
    """
    from storage.index_store import load_index

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_codebase first."
        )

    _, functions, _ = load_index(index_dir)

    # Match file by suffix/partial path
    search = file_path.strip().lower().replace("\\", "/")

    matched_funcs = []
    matched_file_path = None

    for func in functions.values():
        func_file = str(func.file_path).replace("\\", "/")
        try:
            func_rel = str(Path(func.file_path).relative_to(project_path)).replace("\\", "/")
        except ValueError:
            func_rel = func_file

        if (func_rel.lower().endswith(search) or
            func_rel.lower() == search or
            Path(func_rel).name.lower() == Path(search).name.lower() or
            search in func_rel.lower()):
            matched_funcs.append(func)
            matched_file_path = func_rel

    if not matched_funcs:
        # List available files
        all_files = sorted(set(
            str(Path(f.file_path).relative_to(project_path))
            for f in functions.values()
            if Path(f.file_path).is_relative_to(project_path)
        ))
        available = "\n".join(f"  • {f}" for f in all_files[:15])
        return (
            f"No file matching '{file_path}' found in the index.\n\n"
            f"Available files:\n{available}\n"
            + (f"  ... and {len(all_files) - 15} more" if len(all_files) > 15 else "")
        )

    # Sort by line number
    matched_funcs.sort(key=lambda f: f.line_start)

    # Build all qualified names in this file for cross-ref
    file_func_names = {f.qualified_name for f in matched_funcs}

    lines = [
        f"=== FILE OVERVIEW: {matched_file_path} ===",
        f"Functions defined: {len(matched_funcs)}",
        "",
    ]

    for func in matched_funcs:
        sig = func.signature if hasattr(func, 'signature') and func.signature else f"function {func.name}(...)"
        lines.append(f"{'─' * 50}")
        lines.append(f"  {func.name}  (line {func.line_start}–{func.line_end})")
        lines.append(f"  Signature: {sig}")

        if func.docstring:
            doc = func.docstring.strip().split("\n")[0][:120]
            lines.append(f"  Purpose:   {doc}")

        # What this function calls (within this file vs external)
        internal_calls = [c for c in func.calls if c in file_func_names]
        external_calls = [c for c in func.calls if c not in file_func_names]

        if internal_calls:
            lines.append(f"  Calls (internal): {', '.join(internal_calls)}")
        if external_calls:
            lines.append(f"  Calls (external): {', '.join(external_calls[:5])}"
                         + (" ..." if len(external_calls) > 5 else ""))

        # What calls this function (callers from entire codebase)
        callers = [
            f.qualified_name for f in functions.values()
            if func.qualified_name in f.calls and f.qualified_name != func.qualified_name
        ]
        if callers:
            lines.append(f"  Called by: {', '.join(callers[:5])}"
                         + (" ..." if len(callers) > 5 else ""))

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: find_dependents
# ---------------------------------------------------------------------------

@mcp.tool()
def find_dependents(function_name: str, path: str = "") -> str:
    """Find all functions that call or depend on a specific function.

    Call this when the user asks about dependencies or usages, including:
    "what calls X?", "who uses X?", "what depends on X?",
    "where is X called?", "find all callers of X",
    "what breaks if I change X?", "where is X used?".

    Traces both direct callers (functions that call X directly) and
    indirect dependents (functions that call callers of X) using the
    dependency graph — no LLM needed.

    If path is not provided or invalid, automatically uses the current project directory.

    Args:
        function_name: Name of the function to find dependents for.
                      Can be partial (e.g. "compress") or fully qualified.
        path: Absolute path to the indexed project directory. Leave empty to use current project.

    Returns:
        List of direct callers and indirect dependents with file locations.
    """
    from storage.index_store import load_index

    # Fall back to client CWD if model passed nothing or garbage
    resolved = path.strip() if path else ""
    if not resolved or not Path(resolved).exists():
        resolved = _CLIENT_CWD

    project_path = Path(resolved).resolve()
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_codebase first."
        )

    graph, functions, _ = load_index(index_dir)

    # Find matching function(s)
    search_term = function_name.strip().lower()
    targets = []

    if function_name in functions:
        targets.append(functions[function_name])
    else:
        for qname, func in functions.items():
            if search_term in qname.lower() or search_term in func.name.lower():
                targets.append(func)

    if not targets:
        all_names = sorted(functions.keys())
        suggestions = [n for n in all_names if search_term[:3] in n.lower()][:5]
        suggestion_str = "\n".join(f"  • {n}" for n in suggestions) or "  (none found)"
        return (
            f"No function named '{function_name}' found in the index.\n\n"
            f"Similar names:\n{suggestion_str}"
        )

    lines = []

    for target in targets:
        try:
            target_rel = str(Path(target.file_path).relative_to(project_path))
        except ValueError:
            target_rel = str(target.file_path)

        lines.append(f"=== DEPENDENTS OF: {target.qualified_name} ===")
        lines.append(f"Defined in: {target_rel}:{target.line_start}")
        lines.append("")

        # Direct callers — functions whose calls list includes this qualified name
        direct_callers = [
            func for func in functions.values()
            if target.qualified_name in func.calls
            and func.qualified_name != target.qualified_name
        ]

        # Also check graph edges (predecessor nodes call this node)
        if target.qualified_name in graph:
            graph_callers_names = set(graph.predecessors(target.qualified_name))
            for qname in graph_callers_names:
                if qname in functions:
                    func = functions[qname]
                    if func not in direct_callers and func.qualified_name != target.qualified_name:
                        direct_callers.append(func)

        if direct_callers:
            lines.append(f"Direct callers ({len(direct_callers)}):")
            for caller in direct_callers:
                try:
                    caller_rel = str(Path(caller.file_path).relative_to(project_path))
                except ValueError:
                    caller_rel = str(caller.file_path)
                lines.append(f"  → {caller.qualified_name}")
                lines.append(f"    File: {caller_rel}:{caller.line_start}")
        else:
            lines.append("Direct callers: none found")
            lines.append("  (This function may be an entry point, event handler, or called dynamically)")

        # Indirect dependents — who calls the callers?
        indirect = []
        seen = {target.qualified_name} | {c.qualified_name for c in direct_callers}
        for caller in direct_callers:
            indirect_callers = [
                func for func in functions.values()
                if caller.qualified_name in func.calls
                and func.qualified_name not in seen
            ]
            for f in indirect_callers:
                if f.qualified_name not in seen:
                    indirect.append((f, caller))
                    seen.add(f.qualified_name)

        if indirect:
            lines.append("")
            lines.append(f"Indirect dependents ({len(indirect)}):")
            for func, via in indirect[:10]:
                try:
                    func_rel = str(Path(func.file_path).relative_to(project_path))
                except ValueError:
                    func_rel = str(func.file_path)
                lines.append(f"  → {func.qualified_name} (via {via.qualified_name})")
                lines.append(f"    File: {func_rel}:{func.line_start}")
            if len(indirect) > 10:
                lines.append(f"  ... and {len(indirect) - 10} more")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server (stdio transport for Claude Desktop / Cursor)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
