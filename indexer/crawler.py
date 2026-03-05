"""Crawl directories to find Python files for indexing."""

from pathlib import Path


def crawl_directory(directory: Path) -> list[Path]:
    """
    Walk a directory recursively and return all source code files.
    
    Supports: .py, .js, .jsx, .ts, .tsx
    
    Skips:
    - Hidden files and folders (starting with a dot)
    - __pycache__ folders
    - node_modules/, .next/, dist/, build/, .git/
    
    Args:
        directory: The root directory to crawl.
        
    Returns:
        List of Path objects pointing to source code files.
        
    Raises:
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If directory is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    # Directories to skip
    skip_dirs = {
        "__pycache__", "node_modules", ".next", "dist", "build", ".git",
        "out", "coverage", ".nuxt", ".vite", "vendor", ".venv"
    }
    
    # Supported file extensions
    supported_extensions = {".py", ".js", ".jsx", ".ts", ".tsx"}
    
    source_files: list[Path] = []
    
    for path in directory.rglob("*"):
        # Skip hidden files and folders (starting with .)
        if any(part.startswith(".") for part in path.parts):
            continue
        
        # Skip specific directories
        if any(skip_dir in path.parts for skip_dir in skip_dirs):
            continue
        
        # Only include supported file extensions
        if path.is_file() and path.suffix in supported_extensions:
            source_files.append(path)
    
    return source_files
