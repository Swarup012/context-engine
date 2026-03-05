"""Tests for the file watcher."""

import os
import time
from pathlib import Path

import pytest

from indexer.graph_builder import build_graph
from indexer.watcher import IndexWatcher
from storage.index_store import save_index, load_index
from indexer.embedder import generate_embeddings


# Mark watcher tests as slow integration tests
pytestmark = pytest.mark.slow


@pytest.fixture
def watched_project(tmp_path):
    """Create a temporary project with an index to watch."""
    # Use a real directory instead of pytest tmp_path which may not trigger events
    # Use /tmp with a unique name
    import tempfile
    import shutil
    
    watch_dir = Path(tempfile.mkdtemp(prefix='watcher_test_'))
    
    # Create a simple Python file
    test_file = watch_dir / "test.py"
    test_file.write_text("""
def hello():
    '''Say hello'''
    return "hello"

def world():
    '''Say world'''
    return "world"
""")
    
    # Build initial index
    graph, functions = build_graph(watch_dir)
    index_dir = watch_dir / ".context-engine"
    save_index(graph, functions, index_dir)
    generate_embeddings(functions, index_dir, show_progress=False)
    
    yield watch_dir
    
    # Cleanup
    try:
        shutil.rmtree(watch_dir)
    except Exception:
        pass


def test_modify_file_triggers_reindex(watched_project):
    """Test that modifying a file triggers re-indexing of that file only."""
    test_file = watched_project / "test.py"
    
    # Track changes
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    # Start watcher
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Give watcher time to start up and begin watching
        time.sleep(1)
        
        # Modify the file
        test_file.write_text("""
def hello():
    '''Say hello'''
    return "hello world"

def goodbye():
    '''Say goodbye'''
    return "goodbye"
""")
        
        # Force filesystem sync
        test_file.touch()
        
        # Wait for debounce (2s) + processing (1s extra)
        time.sleep(4)
        
        # Should have detected the change
        assert len(changes) > 0, f"No changes detected. Watcher may not be working."
        
        event_type, file_path, stats = changes[0]
        assert event_type == "modified"
        assert "test.py" in file_path
        
        # Should have added goodbye, updated hello, removed world
        assert stats["functions_added"] == 1  # goodbye
        assert stats["functions_updated"] == 1  # hello
        assert stats["functions_removed"] == 1  # world
        
    finally:
        watcher.stop()


def test_debouncing_works(watched_project):
    """Test that debouncing works - 5 rapid saves = 1 re-index."""
    test_file = watched_project / "test.py"
    
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Give watcher time to start
        time.sleep(1)
        
        # Rapidly modify the file 5 times
        for i in range(5):
            test_file.write_text(f"""
def hello():
    '''Say hello {i}'''
    return "hello {i}"
""")
            test_file.touch()  # Force filesystem event
            time.sleep(0.2)  # Small delay between writes
        
        # Wait for debounce (2 seconds) + processing (2s extra)
        time.sleep(4)
        
        # Should have only processed once due to debouncing
        assert len(changes) == 1, f"Expected 1 change due to debouncing, got {len(changes)}"
        
    finally:
        watcher.stop()


def test_delete_file_removes_functions(watched_project):
    """Test that deleting a file removes its functions from the graph."""
    test_file = watched_project / "test.py"
    
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Give watcher time to start
        time.sleep(1)
        
        # Delete the file
        test_file.unlink()
        
        # Wait for debounce + processing
        time.sleep(4)
        
        # Should have detected deletion
        assert len(changes) > 0, f"File deletion not detected. Changes: {changes}"
        
        event_type, file_path, stats = changes[0]
        assert event_type == "deleted"
        assert "test.py" in file_path
        assert stats["functions_removed"] == 2  # hello and world
        
        # Verify functions are removed from index
        index_dir = watched_project / ".context-engine"
        graph, functions, _ = load_index(index_dir)
        
        assert "test.hello" not in functions
        assert "test.world" not in functions
        
    finally:
        watcher.stop()


def test_create_new_file_adds_to_index(watched_project):
    """Test that creating a new file adds it to the index."""
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Give watcher time to start
        time.sleep(1)
        
        # Create a new file
        new_file = watched_project / "new.py"
        new_file.write_text("""
def new_function():
    '''A new function'''
    return "new"
""")
        new_file.touch()  # Force filesystem event
        
        # Wait for debounce + processing
        time.sleep(4)
        
        # Should have detected creation (or modification - watchdog may report either)
        assert len(changes) > 0, f"File creation not detected. Changes: {changes}"
        
        event_type, file_path, stats = changes[0]
        assert event_type in ("created", "modified"), f"Expected created or modified, got {event_type}"
        assert "new.py" in file_path
        assert stats["functions_added"] == 1
        
        # Verify function is in index
        index_dir = watched_project / ".context-engine"
        graph, functions, _ = load_index(index_dir)
        
        assert "new.new_function" in functions
        
    finally:
        watcher.stop()


def test_watcher_ignores_hidden_files(watched_project):
    """Test that watcher ignores hidden files and directories."""
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Create a hidden file
        hidden_file = watched_project / ".hidden.py"
        hidden_file.write_text("def hidden(): pass")
        
        # Wait
        time.sleep(3)
        
        # Should not have triggered any changes
        assert len(changes) == 0
        
    finally:
        watcher.stop()


def test_watcher_ignores_node_modules(watched_project):
    """Test that watcher ignores node_modules directory."""
    changes = []
    
    def on_change(event_type, file_path, stats):
        changes.append((event_type, file_path, stats))
    
    watcher = IndexWatcher(watched_project, on_change=on_change)
    watcher.start()
    
    try:
        # Create node_modules directory with a file
        node_modules = watched_project / "node_modules"
        node_modules.mkdir()
        
        test_file = node_modules / "test.js"
        test_file.write_text("function test() {}")
        
        # Wait
        time.sleep(3)
        
        # Should not have triggered any changes
        assert len(changes) == 0
        
    finally:
        watcher.stop()
