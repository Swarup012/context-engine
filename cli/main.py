"""CLI for ContextEngine using Typer."""

import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from assembler.context_builder import assemble_context, format_context_for_llm
from indexer.embedder import generate_embeddings
from indexer.graph_builder import build_graph
from indexer.watcher import IndexWatcher
from llm.client import get_llm_client
from retriever.semantic_search import semantic_search
from storage.index_store import load_index, save_index

# Load environment variables from .env if it exists (local development)
# For global installs, use system environment variables instead
if Path(".env").exists():
    load_dotenv()

app = typer.Typer(help="ContextEngine - Intelligent context assembly for LLMs")
console = Console()


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to the directory to index"),
) -> None:
    """
    Index a Python codebase by building a dependency graph.
    
    Saves the index to .context-engine/ folder in the target directory.
    """
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)
    
    if not path.is_dir():
        console.print(f"[red]Error:[/red] Path is not a directory: {path}")
        raise typer.Exit(1)
    
    # Convert to absolute path
    path = path.resolve()
    
    console.print(f"[cyan]Indexing:[/cyan] {path}")
    
    # Build the graph with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building dependency graph...", total=None)
        graph, functions = build_graph(path)
        progress.update(task, completed=True)
    
    # Save to .context-engine/
    index_dir = path / ".context-engine"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Saving index...", total=None)
        save_index(graph, functions, index_dir)
        progress.update(task, completed=True)
    
    # Generate embeddings
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=None)
        generate_embeddings(functions, index_dir, show_progress=False)
        progress.update(task, completed=True)
    
    # Add to .gitignore
    gitignore_path = path / ".gitignore"
    gitignore_entry = ".context-engine/"
    
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if gitignore_entry not in content:
            with gitignore_path.open("a") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write(f"{gitignore_entry}\n")
            console.print(f"[green]✓[/green] Added .context-engine/ to .gitignore")
    else:
        gitignore_path.write_text(f"{gitignore_entry}\n")
        console.print(f"[green]✓[/green] Created .gitignore with .context-engine/")
    
    # Print summary
    files_indexed = len(set(func.file_path for func in functions.values()))
    
    console.print("\n[green]✓ Indexing complete![/green]\n")
    
    table = Table(show_header=False)
    table.add_row("Files indexed:", f"[cyan]{files_indexed}[/cyan]")
    table.add_row("Functions found:", f"[cyan]{len(functions)}[/cyan]")
    table.add_row("Edges found:", f"[cyan]{graph.number_of_edges()}[/cyan]")
    table.add_row("Index saved to:", f"[dim]{index_dir}[/dim]")
    
    console.print(table)
    console.print()
    console.print(
        "[dim]💡 Tip: Run[/dim] [cyan]context-engine watch .[/cyan] "
        "[dim]to keep index automatically updated as you code.[/dim]"
    )


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text to search for"),
    path: Path = typer.Option(
        Path.cwd(), 
        "--path", 
        "-p", 
        help="Path to the indexed directory"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results to return"),
) -> None:
    """
    Query the indexed codebase using keyword matching.
    
    Searches function names, qualified names, and docstrings.
    """
    # Convert to absolute path
    path = path.resolve()
    index_dir = path / ".context-engine"
    
    # Check if index exists
    if not index_dir.exists():
        console.print(
            "[red]Error:[/red] No index found. "
            f"Run [cyan]context-engine index {path}[/cyan] first."
        )
        raise typer.Exit(1)
    
    # Perform semantic search
    try:
        results = semantic_search(text, path, top_k=limit)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Display results
    if not results:
        console.print(f"[yellow]No results found for:[/yellow] {text}")
        return
    
    console.print(f"[cyan]Top {len(results)} results for:[/cyan] {text}\n")
    
    for i, (score, func) in enumerate(results, 1):
        console.print(f"[bold]{i}. {func.qualified_name}[/bold]")
        console.print(f"   [dim]File:[/dim] {func.file_path}:{func.line_start}")
        
        if func.docstring:
            # Get first line of docstring
            first_line = func.docstring.split("\n")[0].strip()
            console.print(f"   [dim]Doc:[/dim] {first_line}")
        
        console.print()


@app.command()
def status(
    path: Path = typer.Option(
        Path.cwd(), 
        "--path", 
        "-p", 
        help="Path to the indexed directory"
    ),
) -> None:
    """
    Show status of the current index.
    """
    # Convert to absolute path
    path = path.resolve()
    index_dir = path / ".context-engine"
    
    # Check if index exists
    if not index_dir.exists():
        console.print(
            "[red]Error:[/red] No index found. "
            f"Run [cyan]context-engine index {path}[/cyan] first."
        )
        raise typer.Exit(1)
    
    # Load index
    try:
        graph, functions, metadata = load_index(index_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Display status
    console.print(f"[cyan]Index Status[/cyan] ({path})\n")
    
    table = Table(show_header=False)
    table.add_row("Total files indexed:", f"[cyan]{metadata.get('total_files', 'N/A')}[/cyan]")
    table.add_row("Total functions found:", f"[cyan]{metadata.get('total_functions', 'N/A')}[/cyan]")
    table.add_row("Total edges:", f"[cyan]{metadata.get('total_edges', 'N/A')}[/cyan]")
    
    # Format last indexed time
    last_indexed = metadata.get("last_indexed", "Unknown")
    if last_indexed != "Unknown":
        try:
            dt = datetime.fromisoformat(last_indexed)
            last_indexed = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    
    table.add_row("Last indexed:", f"[dim]{last_indexed}[/dim]")
    
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the codebase"),
    path: Path = typer.Option(
        Path.cwd(),
        "--path",
        "-p",
        help="Path to the indexed directory"
    ),
    budget: int = typer.Option(
        150000,
        "--budget",
        "-b",
        help="Token budget for assembled context"
    ),
) -> None:
    """
    Ask a question about the codebase and get an AI-powered answer.
    
    Assembles relevant context and sends it to an LLM for analysis.
    """
    # Convert to absolute path
    path = path.resolve()
    index_dir = path / ".context-engine"
    
    # Check if index exists, if not run indexing automatically
    if not index_dir.exists():
        console.print("[yellow]No index found. Indexing project first...[/yellow]\n")
        
        # Run indexing
        graph, functions = build_graph(path)
        save_index(graph, functions, index_dir)
        generate_embeddings(functions, index_dir, show_progress=False)
        
        console.print("[green]✓ Indexing complete![/green]\n")
    
    # Check for stale index
    try:
        _, _, metadata = load_index(index_dir)
        
        # Check if any files were modified after last index
        last_indexed_str = metadata.get("last_indexed")
        if last_indexed_str:
            from datetime import datetime
            last_indexed = datetime.fromisoformat(last_indexed_str)
            
            # Check for files modified after last_indexed
            stale_files = []
            for file_path in path.rglob("*"):
                # Skip ignored directories
                if any(part in file_path.parts for part in {".context-engine", "node_modules", "__pycache__", ".git", ".venv"}):
                    continue
                
                # Check supported files
                if file_path.is_file() and file_path.suffix in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime > last_indexed:
                        stale_files.append(file_path)
            
            # Show warning if files changed
            if stale_files:
                console.print(
                    f"[yellow]⚠ {len(stale_files)} file(s) changed since last index.[/yellow] "
                    f"Run [cyan]context-engine index .[/cyan] or [cyan]context-engine watch .[/cyan] "
                    f"for accurate results.\n"
                )
    except Exception:
        # If we can't check, just continue
        pass
    
    # Assemble context
    console.print(f"[cyan]Question:[/cyan] {question}\n")
    
    try:
        assembled = assemble_context(question, path, token_budget=budget)
    except Exception as e:
        console.print(f"[red]Error assembling context:[/red] {e}")
        raise typer.Exit(1)
    
    if not assembled.chunks:
        console.print("[yellow]No relevant context found for your question.[/yellow]")
        raise typer.Exit(0)
    
    # Format context for LLM
    formatted_context = format_context_for_llm(assembled)
    
    # Build system prompt
    system_prompt = """You are an expert software engineer analyzing a codebase.
You will be given relevant code context assembled intelligently from the codebase.
Answer the user's question precisely based on the provided code.
If the answer is not in the provided context, say so clearly."""
    
    # Build messages
    user_message = f"""CODE CONTEXT:

{formatted_context}

USER QUESTION:
{question}

Please answer the question based on the provided code context."""
    
    messages = [{"role": "user", "content": user_message}]
    
    # Get LLM client
    try:
        llm = get_llm_client()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Stream response
    console.print("[bold]Answer:[/bold]\n")
    
    response_text = ""
    try:
        for chunk in llm.stream(messages, system=system_prompt):
            console.print(chunk, end="")
            response_text += chunk
    except Exception as e:
        console.print(f"\n\n[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Print footer
    console.print("\n")
    console.print("[dim]" + "─" * 60 + "[/dim]")
    console.print(
        f"[dim]Context used: {assembled.total_tokens:,} tokens | "
        f"Model: {llm.get_model_name()} | "
        f"Focal point: {assembled.focal_point}[/dim]"
    )


@app.command()
def assemble(
    query: str = typer.Argument(..., help="Query to assemble context for"),
    path: Path = typer.Option(
        Path.cwd(),
        "--path",
        "-p",
        help="Path to the indexed directory"
    ),
    budget: int = typer.Option(
        150000,
        "--budget",
        "-b",
        help="Token budget for assembled context"
    ),
) -> None:
    """
    Assemble context with hot/warm/cold tiers for a query.
    
    Shows the breakdown of functions in each tier with token counts.
    """
    # Convert to absolute path
    path = path.resolve()
    index_dir = path / ".context-engine"
    
    # Check if index exists
    if not index_dir.exists():
        console.print(
            "[red]Error:[/red] No index found. "
            f"Run [cyan]context-engine index {path}[/cyan] first."
        )
        raise typer.Exit(1)
    
    # Assemble context
    console.print(f"[cyan]Assembling context for:[/cyan] {query}\n")
    
    try:
        assembled = assemble_context(query, path, token_budget=budget)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Display results
    if assembled.query_analysis:
        console.print(f"[bold]Query Type:[/bold] {assembled.query_analysis.query_type}")
        
        if len(assembled.query_analysis.focal_points) > 1:
            focal_points_str = ", ".join(assembled.query_analysis.focal_points)
            console.print(f"[bold]Focal Points:[/bold] {focal_points_str}")
        else:
            console.print(f"[bold]Focal Point:[/bold] {assembled.focal_point}")
        
        if assembled.query_analysis.concepts:
            concepts_str = ", ".join(assembled.query_analysis.concepts)
            console.print(f"[bold]Concepts:[/bold] {concepts_str}")
        
        console.print()
    else:
        console.print(f"[bold]Focal Point:[/bold] {assembled.focal_point}\n")
    
    # Count chunks per tier
    hot_chunks = [c for c in assembled.chunks if c.tier == "hot"]
    warm_chunks = [c for c in assembled.chunks if c.tier == "warm"]
    cold_chunks = [c for c in assembled.chunks if c.tier == "cold"]
    
    hot_tokens = sum(c.token_count for c in hot_chunks)
    warm_tokens = sum(c.token_count for c in warm_chunks)
    cold_tokens = sum(c.token_count for c in cold_chunks)
    
    # Summary table
    summary_table = Table(title="Context Assembly Summary")
    summary_table.add_column("Tier", style="cyan")
    summary_table.add_column("Functions", justify="right")
    summary_table.add_column("Tokens", justify="right")
    summary_table.add_column("% of Total", justify="right")
    
    if assembled.total_tokens > 0:
        hot_pct = (hot_tokens / assembled.total_tokens) * 100
        warm_pct = (warm_tokens / assembled.total_tokens) * 100
        cold_pct = (cold_tokens / assembled.total_tokens) * 100
    else:
        hot_pct = warm_pct = cold_pct = 0
    
    summary_table.add_row("HOT", str(len(hot_chunks)), f"{hot_tokens:,}", f"{hot_pct:.1f}%")
    summary_table.add_row("WARM", str(len(warm_chunks)), f"{warm_tokens:,}", f"{warm_pct:.1f}%")
    # COLD row: show how many were filtered out by similarity threshold
    cold_filtered = getattr(assembled, "cold_filtered_count", 0)
    cold_label = f"{len(cold_chunks)} relevant (filtered {cold_filtered} irrelevant)" if cold_filtered > 0 else str(len(cold_chunks))
    summary_table.add_row("COLD", cold_label, f"{cold_tokens:,}", f"{cold_pct:.1f}%")
    summary_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(assembled.chunks)}[/bold]",
        f"[bold]{assembled.total_tokens:,}[/bold]",
        f"[bold]{assembled.budget_used_percent:.1f}%[/bold]"
    )
    
    console.print(summary_table)
    console.print()
    
    # Budget info
    console.print(f"[dim]Token Budget:[/dim] {budget:,}")
    console.print(f"[dim]Tokens Used:[/dim] {assembled.total_tokens:,} ({assembled.budget_used_percent:.1f}%)")
    console.print(f"[dim]Remaining:[/dim] {budget - assembled.total_tokens:,}\n")
    
    # List functions in each tier
    if hot_chunks:
        console.print("[bold red]HOT Functions (full source):[/bold red]")
        for chunk in hot_chunks:
            console.print(f"  • {chunk.node.qualified_name} ({chunk.token_count} tokens)")
        console.print()
    
    if warm_chunks:
        console.print("[bold yellow]WARM Functions (LLM-compressed summaries):[/bold yellow]")
        for chunk in warm_chunks:
            cached_label = "[dim](cached)[/dim]" if chunk.was_cached else "[dim](freshly compressed)[/dim]"
            console.print(f"  • {chunk.node.qualified_name} ({chunk.token_count} tokens) {cached_label}")
        console.print()
    
    if cold_chunks:
        cold_filtered = getattr(assembled, "cold_filtered_count", 0)
        cold_header = (
            f"[bold blue]COLD Functions — {len(cold_chunks)} relevant "
            f"(filtered {cold_filtered} irrelevant, threshold ≥0.3):[/bold blue]"
            if cold_filtered > 0
            else "[bold blue]COLD Functions (signature only):[/bold blue]"
        )
        console.print(cold_header)
        for chunk in cold_chunks:
            score_label = f" [dim][score: {chunk.relevance_score:.2f}][/dim]"
            console.print(f"  • {chunk.node.qualified_name} ({chunk.token_count} tokens){score_label}")
        console.print()


@app.command()
def watch(
    path: Path = typer.Argument(
        None,
        help="Path to the directory to watch (default: current directory)"
    ),
) -> None:
    """
    Watch a project directory and keep the index automatically updated.
    
    Monitors file changes and incrementally updates the index in real-time.
    Press Ctrl+C to stop watching.
    """
    # Default to current directory if not provided
    if path is None:
        path = Path.cwd()
    
    # Convert to absolute path
    path = path.resolve()
    index_dir = path / ".context-engine"
    
    # Check if index exists
    if not index_dir.exists():
        console.print(
            "[yellow]No index found. Creating initial index...[/yellow]\n"
        )
        
        # Build initial index
        graph, functions = build_graph(path)
        save_index(graph, functions, index_dir)
        generate_embeddings(functions, index_dir, show_progress=False)
        
        console.print("[green]✓ Initial index created![/green]\n")
    
    console.print(f"[cyan]Watching:[/cyan] {path}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    
    # Track stats
    last_update_msg = "Index up to date"
    last_update_time = time.time()
    
    def on_change(event_type: str, file_path: str, stats: dict) -> None:
        """Callback when files change."""
        nonlocal last_update_msg, last_update_time
        
        # Build status message
        parts = []
        if stats.get("functions_added", 0) > 0:
            parts.append(f"+{stats['functions_added']}")
        if stats.get("functions_updated", 0) > 0:
            parts.append(f"~{stats['functions_updated']}")
        if stats.get("functions_removed", 0) > 0:
            parts.append(f"-{stats['functions_removed']}")
        
        stats_str = " ".join(parts) if parts else "no changes"
        last_update_msg = f"Updated {file_path} ({stats_str})"
        last_update_time = time.time()
    
    # Create watcher
    watcher = IndexWatcher(path, on_change=on_change)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        console.print("\n\n[yellow]Stopping watcher...[/yellow]")
        watcher.stop()
        console.print("[green]✓ Watcher stopped. Index is up to date.[/green]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start watching
    watcher.start()
    
    # Live status display
    try:
        while True:
            # Calculate time since last update
            seconds_ago = int(time.time() - last_update_time)
            
            if seconds_ago == 0:
                time_str = "just now"
            elif seconds_ago == 1:
                time_str = "1 second ago"
            elif seconds_ago < 60:
                time_str = f"{seconds_ago} seconds ago"
            elif seconds_ago < 120:
                time_str = "1 minute ago"
            else:
                minutes = seconds_ago // 60
                time_str = f"{minutes} minutes ago"
            
            # Clear line and print status
            console.print(
                f"\r[green]●[/green] {last_update_msg} — last update: {time_str}",
                end="",
                markup=True
            )
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Stopping watcher...[/yellow]")
        watcher.stop()
        console.print("[green]✓ Watcher stopped. Index is up to date.[/green]")


if __name__ == "__main__":
    app()
