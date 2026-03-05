"""Build a dependency graph from a codebase using NetworkX."""

import logging
from pathlib import Path

import networkx as nx

from indexer.crawler import crawl_directory
from indexer.parser import parse_file
from models import FunctionNode

logger = logging.getLogger(__name__)


def build_graph(directory: Path) -> tuple[nx.DiGraph, dict[str, FunctionNode]]:
    """
    Build a dependency graph from all Python files in a directory.
    
    Creates a directed graph where:
    - Nodes are function qualified_names (e.g., "auth.validate_token")
    - Each node stores the full FunctionNode as metadata
    - Edges represent "caller -> callee" relationships
    
    Args:
        directory: Root directory to crawl for Python files.
        
    Returns:
        A tuple containing:
        - NetworkX DiGraph with function call dependencies
        - Dictionary mapping qualified_name -> FunctionNode
        
    Raises:
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If directory is not a directory.
    """
    # Get all Python files
    python_files = crawl_directory(directory)
    logger.info(f"Found {len(python_files)} Python files in {directory}")
    
    # Parse all files and collect FunctionNodes
    all_functions: dict[str, FunctionNode] = {}
    
    for file_path in python_files:
        try:
            functions = parse_file(file_path)
            for func in functions:
                all_functions[func.qualified_name] = func
            logger.debug(f"Parsed {len(functions)} functions from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            # Continue processing other files
            continue
    
    logger.info(f"Parsed {len(all_functions)} total functions")
    
    # Build the graph
    graph = nx.DiGraph()
    
    # Add all functions as nodes
    for qualified_name, func_node in all_functions.items():
        graph.add_node(qualified_name, function=func_node)
    
    # Add edges based on function calls
    for qualified_name, func_node in all_functions.items():
        for call in func_node.calls:
            # Try to resolve the call to a qualified name
            # The call might be just a function name or a qualified name
            callee_qualified_name = _resolve_call(call, func_node, all_functions)
            
            if callee_qualified_name and callee_qualified_name in all_functions:
                # Add edge: caller -> callee
                graph.add_edge(qualified_name, callee_qualified_name)
                logger.debug(f"Added edge: {qualified_name} -> {callee_qualified_name}")
            else:
                # Call to external function or unresolved - skip
                logger.debug(f"Unresolved call from {qualified_name}: {call}")
    
    logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Check for cycles (for informational purposes)
    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        logger.info(f"Graph contains {len(cycles)} cycles")
    
    return graph, all_functions


def _resolve_call(
    call: str,
    caller_func: FunctionNode,
    all_functions: dict[str, FunctionNode]
) -> str | None:
    """
    Resolve a function call to its qualified name.
    
    Args:
        call: The function call string (e.g., "validate_token" or "auth.validate_token")
        caller_func: The FunctionNode making the call
        all_functions: Dictionary of all known functions
        
    Returns:
        The qualified name if resolved, None otherwise.
    """
    # If the call is already a qualified name and exists, use it
    if call in all_functions:
        return call
    
    # Try to find a function with this name in the same module
    # Get the module name from the caller's qualified name
    # e.g., "auth.validate_token" -> "auth"
    caller_parts = caller_func.qualified_name.split(".")
    if len(caller_parts) >= 2:
        module_name = ".".join(caller_parts[:-1])
        potential_qualified_name = f"{module_name}.{call}"
        if potential_qualified_name in all_functions:
            return potential_qualified_name
    
    # Try to find by matching just the function name
    # This handles cases where the call is just the function name
    for qualified_name, func in all_functions.items():
        if func.name == call:
            return qualified_name
    
    # Could not resolve
    return None
