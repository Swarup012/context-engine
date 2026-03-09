import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


if Path(".env").exists():
    load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# _CLIENT_CWD = os.environ.get("PWD") or os.getcwd()
ACTIVE_PROJECT_FILE = Path.home() / ".context-engine-active"

def resolve_project_path(path: str) -> Path:
    """
    Resolve the project directory consistently across MCP tools.

    Priority:
    1. Explicit path passed to tool
    2. Current working directory if it contains a .context-engine index
    3. Last indexed project stored in ~/.context-engine-active
    """

    
    if path and path.strip():
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        return resolved

    
    cwd = Path.cwd()
    if (cwd / ".context-engine").exists():
        return cwd.resolve()

    
    if ACTIVE_PROJECT_FILE.exists():
        stored = ACTIVE_PROJECT_FILE.read_text().strip()
        stored_path = Path(stored)
        if stored_path.exists():
            return stored_path.resolve()

    raise ValueError(
        "No project found. Run index_codebase() first."
    )


mcp = FastMCP(
    "ContextEngine",
    instructions="""
ContextEngine provides structured context from an indexed codebase.

Primary tool: analyze_codebase

Rules:
1. ALWAYS call analyze_codebase when answering questions about code, architecture, bugs, or how something works.
2. If the project is not indexed yet, call index_project first.
3. Use find_code to locate functions, routes, commands, or concepts.
4. Use show_function_source to display the full source code of a function.
5. Use explain_file_structure to understand what a file does.
6. Use find_callers to see which functions depend on another function.
7. Use check_index_status to verify whether the codebase index is up to date.

Never answer code questions from memory — always use these tools first.

The project directory is detected automatically unless a path is explicitly provided.
""",
)




@mcp.tool(
    name="index_project",
    description="Index a project so ContextEngine can understand the codebase."
)
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

    project_path = resolve_project_path(path)

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

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
    ACTIVE_PROJECT_FILE.write_text(str(project_path))

    # Compute summary stats
    files_indexed = len(set(str(func.file_path) for func in functions.values()))
    total_functions = len(functions)
    total_edges = graph.number_of_edges()

    return (
        f"Indexed {files_indexed} files, {total_functions} functions, {total_edges} edges. "
        f"Index saved to {index_dir}"
    )




@mcp.tool(
    name="analyze_codebase",
    description="""
Primary tool for understanding a codebase.

Use this for questions like:
- "how does authentication work?"
- "why does login fail?"
- "explain the architecture"
- "where is OAuth implemented?"
- "how does caching interact with auth?"

This tool retrieves the most relevant code using ContextEngine's graph traversal and semantic search.
"""
)
def ask_codebase(
    query: str | None = None,
    question: str | None = None,
    prompt: str | None = None,
    path: str = "",
    token_budget: int = 150000
) -> str:
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
    query = query or question or prompt

    if not query:
        raise ValueError(
            "Missing query parameter. Provide 'query', 'question', or 'prompt'."
        )
    query = query.strip()
    
    from assembler.context_builder import assemble_context, format_context_for_llm

    # Fall back to client CWD if model passed nothing or garbage
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            "Run index_project() first."
        )

    # Run full assembly pipeline
    # Use heuristic approach for MCP: no LLM calls, faster and free
    assembled = assemble_context(query, project_path, token_budget=token_budget, use_llm=False)

    if not assembled.chunks:
        return (
            f"No relevant context found for query: {query}\n\n"
            "The index may be empty or the query doesn't match any indexed functions. "
            "Try re-indexing with index_project()"
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




@mcp.tool(
    name="check_index_status",
    description="""
Check whether a project has been indexed.

Examples:
- "is the project indexed?"
- "how many files are indexed?"
- "when was the index last updated?"
"""
)
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
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        return (
            f"No index found for: {path}\n"
            "Run index_project() to create an index."
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
        lines.append("Run index_project() to refresh the index.")
    else:
        lines.append("✓ Index is up to date.")

    return "\n".join(lines)




@mcp.tool(
    name="find_code",
    description="""
Search the codebase for functions or concepts.

Examples:
- "find auth middleware"
- "where is rate limiting implemented"
- "find payment service"
- "show login handler"
"""
)
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
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_project() first."
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
        lines.append(f"   File: {rel_path}:{func.line_start}")
        lines.append(f"   Signature: {sig}")
        lines.append(f"   Relevance: {score:.0%}")
        if func.docstring:
            # First line of docstring only
            doc_first = func.docstring.strip().split("\n")[0][:120]
            lines.append(f"   Doc: {doc_first}")
        lines.append("")

    return "\n".join(lines)




@mcp.tool(
    name="show_function_source",
    description="""
Show the full source code of a function.

Examples:
- "show validateToken"
- "open compressCommand function"
- "display code for authenticateUser"
"""
)
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
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_project() first."
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




@mcp.tool(
    name="explain_file_structure",
    description="""
Explain the structure of a file.

Examples:
- "explain auth.py"
- "what does database.ts do?"
- "show structure of middleware file"
"""
)
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
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_project() first."
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




@mcp.tool(
     name="find_callers",
    description="""
Find which functions call another function.

Examples:
- "who calls validateToken"
- "where is authenticateUser used"
- "what depends on checkExpiry"
- "find callers of compressCommand"
"""
)
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
    project_path = resolve_project_path(path)
    index_dir = project_path / ".context-engine"

    if not project_path.exists():
        raise ValueError(f"Path does not exist: {project_path}")

    if not index_dir.exists():
        raise ValueError(
            f"No index found at {index_dir}. "
            f"Run index_project() first."
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




def main() -> None:
    """Run the MCP server (stdio transport for Claude Desktop / Cursor)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
