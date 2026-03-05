"""Generate and store embeddings for functions using ChromaDB."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from models import FunctionNode

logger = logging.getLogger(__name__)


def generate_embeddings(
    functions: dict[str, FunctionNode],
    index_dir: Path,
    show_progress: bool = True
) -> None:
    """
    Generate embeddings for functions and store in ChromaDB.
    
    Args:
        functions: Dictionary mapping qualified_name -> FunctionNode.
        index_dir: Directory containing the index (e.g., .context-engine/).
        show_progress: Whether to show progress indicator.
    """
    # Create chroma directory
    chroma_dir = index_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    collection_name = "functions"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedding model (CPU only)
    if show_progress:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Get existing IDs to check what needs updating
    existing_ids = set()
    try:
        existing_data = collection.get()
        if existing_data and existing_data['ids']:
            existing_ids = set(existing_data['ids'])
    except Exception:
        # Collection might be empty
        pass
    
    # Prepare data for embedding
    ids_to_add = []
    documents_to_add = []
    metadatas_to_add = []
    
    ids_to_update = []
    documents_to_update = []
    metadatas_to_update = []
    
    for qualified_name, func in functions.items():
        # Combine source code and docstring for embedding
        doc_text = func.source_code
        if func.docstring:
            doc_text = f"{func.docstring}\n\n{func.source_code}"
        
        metadata = {
            "qualified_name": func.qualified_name,
            "file_path": str(func.file_path),
            "line_start": func.line_start,
            "line_end": func.line_end,
        }
        
        # Check if this function already exists
        if qualified_name in existing_ids:
            ids_to_update.append(qualified_name)
            documents_to_update.append(doc_text)
            metadatas_to_update.append(metadata)
        else:
            ids_to_add.append(qualified_name)
            documents_to_add.append(doc_text)
            metadatas_to_add.append(metadata)
    
    # Generate embeddings and add/update
    total_ops = len(ids_to_add) + len(ids_to_update)
    
    if show_progress:
        logger.info(f"Generating embeddings for {total_ops} functions...")
    
    # Add new functions
    if ids_to_add:
        embeddings = model.encode(documents_to_add, show_progress_bar=show_progress)
        collection.add(
            ids=ids_to_add,
            embeddings=embeddings.tolist(),
            documents=documents_to_add,
            metadatas=metadatas_to_add
        )
        if show_progress:
            logger.info(f"Added {len(ids_to_add)} new functions")
    
    # Update existing functions
    if ids_to_update:
        embeddings = model.encode(documents_to_update, show_progress_bar=show_progress)
        collection.update(
            ids=ids_to_update,
            embeddings=embeddings.tolist(),
            documents=documents_to_update,
            metadatas=metadatas_to_update
        )
        if show_progress:
            logger.info(f"Updated {len(ids_to_update)} existing functions")
    
    # Remove functions that no longer exist
    ids_to_remove = existing_ids - set(functions.keys())
    if ids_to_remove:
        collection.delete(ids=list(ids_to_remove))
        if show_progress:
            logger.info(f"Removed {len(ids_to_remove)} deleted functions")
    
    if show_progress:
        logger.info(f"Embeddings stored in {chroma_dir}")
