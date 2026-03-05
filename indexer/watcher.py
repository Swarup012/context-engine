"""File watcher for automatic index updates."""

import logging
import time
from pathlib import Path
from threading import Timer
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class DebouncedEventHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing.
    
    Debounces rapid file changes to avoid re-indexing on every keystroke.
    If a file is saved multiple times within the debounce window, only
    process it once after the window expires.
    """
    
    def __init__(
        self,
        callback: Callable[[str, str], None],
        debounce_seconds: float = 2.0,
        project_path: Path = Path(".")
    ):
        """
        Initialize debounced event handler.
        
        Args:
            callback: Function to call with (event_type, file_path) after debounce.
            debounce_seconds: Seconds to wait after last event before processing.
            project_path: Root path of the project being watched.
        """
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.project_path = project_path.resolve()
        
        # Map of file_path -> Timer
        self.pending_events: dict[str, Timer] = {}
        
        # Map of file_path -> event_type (to remember the last event)
        self.event_types: dict[str, str] = {}
        
        # Directories to skip
        self.skip_dirs = {
            ".context-engine", "node_modules", "__pycache__",
            ".git", ".next", "dist", "build", ".venv"
        }
        
        # Supported extensions
        self.supported_extensions = {".py", ".js", ".jsx", ".ts", ".tsx"}
    
    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        file_path = Path(path)
        
        # Ignore hidden files and folders
        if any(part.startswith(".") for part in file_path.parts):
            # Allow .py, .js, etc. files but not .hidden folders
            if not file_path.is_file():
                return True
            if file_path.name.startswith("."):
                return True
        
        # Ignore specific directories
        if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
            return True
        
        # Only process supported file types
        if file_path.is_file() and file_path.suffix not in self.supported_extensions:
            return True
        
        return False
    
    def _debounce_event(self, event_type: str, file_path: str) -> None:
        """
        Debounce an event.
        
        If the same file has a pending timer, cancel it and start a new one.
        This ensures we only process the file once after it stops changing.
        """
        # Cancel existing timer for this file
        if file_path in self.pending_events:
            self.pending_events[file_path].cancel()
        
        # Remember the event type
        self.event_types[file_path] = event_type
        
        # Create new timer
        timer = Timer(self.debounce_seconds, self._process_event, args=[file_path])
        self.pending_events[file_path] = timer
        timer.start()
    
    def _process_event(self, file_path: str) -> None:
        """Process a debounced event."""
        event_type = self.event_types.get(file_path, "modified")
        
        # Remove from pending
        if file_path in self.pending_events:
            del self.pending_events[file_path]
        if file_path in self.event_types:
            del self.event_types[file_path]
        
        # Call the callback
        try:
            self.callback(event_type, file_path)
        except Exception as e:
            logger.error(f"Error processing {event_type} event for {file_path}: {e}")
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return
        
        self._debounce_event("modified", event.src_path)
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return
        
        self._debounce_event("created", event.src_path)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if event.is_directory or self._should_ignore(event.src_path):
            return
        
        self._debounce_event("deleted", event.src_path)


class IndexWatcher:
    """
    Watch a project directory and keep the index up to date.
    """
    
    def __init__(self, project_path: Path, on_change: Callable[[str, str, dict], None]):
        """
        Initialize the index watcher.
        
        Args:
            project_path: Path to the project to watch.
            on_change: Callback(event_type, file_path, stats) called after processing changes.
        """
        self.project_path = project_path.resolve()
        self.on_change = on_change
        self.observer: Observer | None = None
        self.last_update_time = time.time()
    
    def start(self) -> None:
        """Start watching the project directory."""
        event_handler = DebouncedEventHandler(
            callback=self._handle_change,
            debounce_seconds=2.0,
            project_path=self.project_path
        )
        
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.project_path), recursive=True)
        self.observer.start()
        logger.info(f"Watching {self.project_path}")
    
    def stop(self) -> None:
        """Stop watching the project directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Watcher stopped")
    
    def _handle_change(self, event_type: str, file_path: str) -> None:
        """
        Handle a file change by updating the index.
        
        Args:
            event_type: Type of event (modified, created, deleted).
            file_path: Absolute path to the changed file.
        """
        # Import at module level to avoid import deadlocks in threads
        import indexer.parser as parser_module
        import storage.index_store as index_store_module
        
        file_path_obj = Path(file_path)
        relative_path = file_path_obj.relative_to(self.project_path)
        
        index_dir = self.project_path / ".context-engine"
        
        try:
            # Load existing index
            graph, functions, metadata = index_store_module.load_index(index_dir)
            
            stats = {"functions_added": 0, "functions_updated": 0, "functions_removed": 0}
            
            if event_type == "deleted":
                # Remove all functions from this file
                functions_to_remove = [
                    qname for qname, func in functions.items()
                    if func.file_path == file_path_obj
                ]
                
                for qname in functions_to_remove:
                    # Remove from functions dict
                    del functions[qname]
                    
                    # Remove from graph
                    if graph.has_node(qname):
                        graph.remove_node(qname)
                
                stats["functions_removed"] = len(functions_to_remove)
                
                logger.info(f"Removed {len(functions_to_remove)} functions from {relative_path}")
            
            elif event_type in ("modified", "created"):
                # Parse the file
                new_functions = parser_module.parse_file(file_path_obj)
                
                # Find old functions from this file
                old_function_names = {
                    func.qualified_name for func in functions.values()
                    if func.file_path == file_path_obj
                }
                
                new_function_names = {func.qualified_name for func in new_functions}
                
                # Remove old functions that no longer exist
                removed_names = old_function_names - new_function_names
                for qname in removed_names:
                    del functions[qname]
                    if graph.has_node(qname):
                        graph.remove_node(qname)
                    stats["functions_removed"] += 1
                
                # Add/update new functions
                for func in new_functions:
                    if func.qualified_name in old_function_names:
                        stats["functions_updated"] += 1
                    else:
                        stats["functions_added"] += 1
                    
                    functions[func.qualified_name] = func
                    
                    # Update graph node
                    if not graph.has_node(func.qualified_name):
                        graph.add_node(func.qualified_name, function=func)
                    else:
                        graph.nodes[func.qualified_name]["function"] = func
                
                # Rebuild edges for affected functions
                # Remove old edges
                for qname in new_function_names:
                    if graph.has_node(qname):
                        # Remove outgoing edges
                        edges_to_remove = list(graph.out_edges(qname))
                        graph.remove_edges_from(edges_to_remove)
                
                # Add new edges based on calls
                for func in new_functions:
                    for call in func.calls:
                        # Try to resolve the call
                        callee_qname = _resolve_call(call, func, functions)
                        if callee_qname and callee_qname in functions:
                            graph.add_edge(func.qualified_name, callee_qname)
                
                logger.info(
                    f"Updated {relative_path}: "
                    f"+{stats['functions_added']} "
                    f"~{stats['functions_updated']} "
                    f"-{stats['functions_removed']} functions"
                )
            
            # Save updated index
            index_store_module.save_index(graph, functions, index_dir)
            
            # NOTE: We intentionally do NOT update embeddings here because:
            # 1. It's slow (can take seconds per file)
            # 2. It causes threading/deadlock issues when called from Timer threads
            # 3. The graph/function index is the critical part for call relationships
            # Users should re-run `context-engine index` to refresh embeddings if needed.
            
            self.last_update_time = time.time()
            
            # Call the callback
            logger.debug(f"About to call on_change callback with: {event_type}, {str(relative_path)}, {stats}")
            self.on_change(event_type, str(relative_path), stats)
            logger.debug(f"on_change callback completed")
            
        except Exception as e:
            import traceback
            logger.error(f"Error updating index for {relative_path}: {e}")
            logger.error(traceback.format_exc())


def _resolve_call(call: str, caller_func, all_functions: dict) -> str | None:
    """Resolve a function call to its qualified name (same logic as graph_builder.py)."""
    if call in all_functions:
        return call
    
    caller_parts = caller_func.qualified_name.split(".")
    if len(caller_parts) >= 2:
        module_name = ".".join(caller_parts[:-1])
        potential_qualified_name = f"{module_name}.{call}"
        if potential_qualified_name in all_functions:
            return potential_qualified_name
    
    for qualified_name, func in all_functions.items():
        if func.name == call:
            return qualified_name
    
    return None
