"""LLM-based function compression for WARM tier context."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from llm.adapters.base import BaseLLMAdapter
from models import FunctionNode

logger = logging.getLogger(__name__)


COMPRESSION_SYSTEM_PROMPT = """You are a code summarizer. Given a function's source code, produce a concise 2-3 sentence technical summary that captures:
1. What the function does
2. Its key parameters and return value
3. Any important side effects or dependencies

Be precise and technical. No fluff."""


def _hash_source(source_code: str) -> str:
    """Generate a hash of the function source code for caching."""
    return hashlib.sha256(source_code.encode('utf-8')).hexdigest()[:16]


def load_compression_cache(index_dir: Path) -> dict[str, str]:
    """
    Load the compression cache from disk.
    
    Args:
        index_dir: Path to .context-engine/ directory.
        
    Returns:
        Dictionary mapping source_hash -> compressed_summary.
    """
    cache_file = index_dir / "compressions.json"
    
    if not cache_file.exists():
        return {}
    
    try:
        with cache_file.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load compression cache: {e}")
        return {}


def save_compression_cache(cache: dict[str, str], index_dir: Path) -> None:
    """
    Save the compression cache to disk.
    
    Args:
        cache: Dictionary mapping source_hash -> compressed_summary.
        index_dir: Path to .context-engine/ directory.
    """
    cache_file = index_dir / "compressions.json"
    
    try:
        with cache_file.open("w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save compression cache: {e}")


def invalidate_cache_for_file(file_path: Path, functions: dict[str, FunctionNode], index_dir: Path) -> None:
    """
    Invalidate cached compressions for functions in a specific file.
    
    Args:
        file_path: Path to the file that changed.
        functions: Dictionary of all functions.
        index_dir: Path to .context-engine/ directory.
    """
    cache = load_compression_cache(index_dir)
    
    # Find functions in this file
    file_functions = [func for func in functions.values() if func.file_path == file_path]
    
    # Remove their cached compressions
    removed_count = 0
    for func in file_functions:
        source_hash = _hash_source(func.source_code)
        if source_hash in cache:
            del cache[source_hash]
            removed_count += 1
    
    if removed_count > 0:
        save_compression_cache(cache, index_dir)
        logger.info(f"Invalidated {removed_count} cached compressions for {file_path}")


def compress_function(
    node: FunctionNode,
    adapter: BaseLLMAdapter,
    index_dir: Path,
    use_cache: bool = True
) -> tuple[str, bool]:
    """
    Compress a function using LLM with caching.
    
    Args:
        node: The FunctionNode to compress.
        adapter: LLM adapter to use for compression.
        index_dir: Path to .context-engine/ directory.
        use_cache: Whether to use cached compressions.
        
    Returns:
        Tuple of (compressed_summary, was_cached).
    """
    source_hash = _hash_source(node.source_code)
    
    # Check cache first
    if use_cache:
        cache = load_compression_cache(index_dir)
        if source_hash in cache:
            logger.debug(f"Using cached compression for {node.qualified_name}")
            return cache[source_hash], True
    
    # Compress with LLM
    try:
        logger.debug(f"Compressing {node.qualified_name} with LLM...")
        
        messages = [
            {"role": "user", "content": f"Function to summarize:\n\n{node.source_code}"}
        ]
        
        summary = adapter.complete(messages, system=COMPRESSION_SYSTEM_PROMPT)
        
        # Cache the result
        if use_cache:
            cache = load_compression_cache(index_dir)
            cache[source_hash] = summary
            save_compression_cache(cache, index_dir)
        
        logger.debug(f"Compressed {node.qualified_name} successfully")
        return summary, False
        
    except Exception as e:
        logger.warning(f"LLM compression failed for {node.qualified_name}: {e}, falling back to truncation")
        
        # Fallback: first 5 lines + docstring
        fallback_parts = []
        if node.docstring:
            fallback_parts.append(f'"""{node.docstring}"""')
        
        source_lines = node.source_code.split('\n')[:5]
        fallback_parts.extend(source_lines)
        
        return '\n'.join(fallback_parts), False


def compress_functions_parallel(
    nodes: list[FunctionNode],
    adapter: BaseLLMAdapter,
    index_dir: Path,
    use_cache: bool = True
) -> dict[str, tuple[str, bool]]:
    """
    Compress multiple functions in parallel (using threading for I/O-bound LLM calls).
    
    Args:
        nodes: List of FunctionNodes to compress.
        adapter: LLM adapter to use.
        index_dir: Path to .context-engine/ directory.
        use_cache: Whether to use cached compressions.
        
    Returns:
        Dictionary mapping qualified_name -> (compressed_summary, was_cached).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    
    # Use ThreadPoolExecutor for I/O-bound LLM calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all compression tasks
        future_to_node = {
            executor.submit(compress_function, node, adapter, index_dir, use_cache): node
            for node in nodes
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_node):
            node = future_to_node[future]
            try:
                compressed, was_cached = future.result()
                results[node.qualified_name] = (compressed, was_cached)
            except Exception as e:
                logger.error(f"Failed to compress {node.qualified_name}: {e}")
                # Use fallback
                fallback_parts = []
                if node.docstring:
                    fallback_parts.append(f'"""{node.docstring}"""')
                source_lines = node.source_code.split('\n')[:5]
                fallback_parts.extend(source_lines)
                results[node.qualified_name] = ('\n'.join(fallback_parts), False)
    
    return results
