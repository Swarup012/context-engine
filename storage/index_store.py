"""Persist and load index data to/from disk."""

import json
from datetime import datetime
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from models import FunctionNode


def save_index(
    graph: nx.DiGraph,
    functions: dict[str, FunctionNode],
    index_dir: Path
) -> None:
    """
    Save the index (graph and functions) to disk.
    
    Args:
        graph: The NetworkX DiGraph of function dependencies.
        functions: Dictionary mapping qualified_name -> FunctionNode.
        index_dir: Directory to save the index files (e.g., .context-engine/).
    """
    # Create index directory if it doesn't exist
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Save functions as JSON
    functions_file = index_dir / "functions.json"
    functions_data = {
        qualified_name: {
            "name": func.name,
            "qualified_name": func.qualified_name,
            "file_path": str(func.file_path),
            "line_start": func.line_start,
            "line_end": func.line_end,
            "source_code": func.source_code,
            "docstring": func.docstring,
            "calls": func.calls,
            "imports": func.imports,
        }
        for qualified_name, func in functions.items()
    }
    
    with functions_file.open("w") as f:
        json.dump(functions_data, f, indent=2)
    
    # Save graph as JSON (using NetworkX's json_graph)
    graph_file = index_dir / "graph.json"
    # Create a copy of the graph without the function metadata (too large)
    # Just store the structure
    graph_data = json_graph.node_link_data(graph)
    # Remove function metadata to keep file size small
    for node in graph_data.get("nodes", []):
        if "function" in node:
            del node["function"]
    
    with graph_file.open("w") as f:
        json.dump(graph_data, f, indent=2)
    
    # Save metadata
    metadata_file = index_dir / "metadata.json"
    metadata = {
        "last_indexed": datetime.now().isoformat(),
        "total_files": len(set(func.file_path for func in functions.values())),
        "total_functions": len(functions),
        "total_edges": graph.number_of_edges(),
    }
    
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)


def load_index(index_dir: Path) -> tuple[nx.DiGraph, dict[str, FunctionNode], dict]:
    """
    Load the index (graph and functions) from disk.
    
    Args:
        index_dir: Directory containing the index files.
        
    Returns:
        A tuple containing:
        - NetworkX DiGraph
        - Dictionary mapping qualified_name -> FunctionNode
        - Metadata dictionary
        
    Raises:
        FileNotFoundError: If index directory or required files don't exist.
    """
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    # Load functions
    functions_file = index_dir / "functions.json"
    if not functions_file.exists():
        raise FileNotFoundError(f"Functions file not found: {functions_file}")
    
    with functions_file.open("r") as f:
        functions_data = json.load(f)
    
    functions = {
        qualified_name: FunctionNode(
            name=data["name"],
            qualified_name=data["qualified_name"],
            file_path=Path(data["file_path"]),
            line_start=data["line_start"],
            line_end=data["line_end"],
            source_code=data["source_code"],
            docstring=data["docstring"],
            calls=data["calls"],
            imports=data["imports"],
        )
        for qualified_name, data in functions_data.items()
    }
    
    # Load graph
    graph_file = index_dir / "graph.json"
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    with graph_file.open("r") as f:
        graph_data = json.load(f)
    
    graph = json_graph.node_link_graph(graph_data, directed=True)
    
    # Re-attach function metadata to nodes
    for node in graph.nodes():
        if node in functions:
            graph.nodes[node]["function"] = functions[node]
    
    # Load metadata
    metadata_file = index_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with metadata_file.open("r") as f:
            metadata = json.load(f)
    
    return graph, functions, metadata
