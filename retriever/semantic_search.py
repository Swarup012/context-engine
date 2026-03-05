"""Semantic search using ChromaDB embeddings."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from models import FunctionNode
from storage.index_store import load_index

logger = logging.getLogger(__name__)


def semantic_search(
    query: str,
    project_path: Path,
    top_k: int = 5
) -> list[tuple[float, FunctionNode]]:
    """
    Search for functions using semantic similarity.
    
    Args:
        query: The search query string.
        project_path: Path to the indexed project.
        top_k: Number of top results to return.
        
    Returns:
        List of tuples (similarity_score, FunctionNode) sorted by relevance.
        
    Raises:
        FileNotFoundError: If ChromaDB collection does not exist.
    """
    index_dir = project_path / ".context-engine"
    chroma_dir = index_dir / "chroma"
    
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB collection not found at {chroma_dir}. "
            f"Run 'context-engine index {project_path}' first."
        )
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get collection
    try:
        collection = client.get_collection(name="functions")
    except Exception as e:
        raise FileNotFoundError(
            f"ChromaDB collection 'functions' not found. "
            f"Run 'context-engine index {project_path}' first."
        ) from e
    
    # Load functions to get full FunctionNode objects
    _, functions, _ = load_index(index_dir)
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Generate query embedding
    query_embedding = model.encode(query, show_progress_bar=False)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, len(functions))
    )
    
    # Build result list with scores and FunctionNodes
    search_results = []
    
    if results and results['ids'] and len(results['ids']) > 0:
        ids = results['ids'][0]
        distances = results['distances'][0] if results['distances'] else [0] * len(ids)
        
        for qualified_name, distance in zip(ids, distances):
            if qualified_name in functions:
                # Convert distance to similarity score (lower distance = higher similarity)
                # For cosine distance: similarity = 1 - distance
                similarity = 1.0 - distance
                search_results.append((similarity, functions[qualified_name]))
    
    return search_results
