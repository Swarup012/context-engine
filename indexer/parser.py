"""Parse source files using tree-sitter to extract function metadata."""

import logging
from pathlib import Path

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser

from models import FunctionNode

logger = logging.getLogger(__name__)

# Initialize tree-sitter languages
PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())


def parse_file(file_path: Path) -> list[FunctionNode]:
    """
    Parse a source file and extract all function definitions.
    
    Supports: Python (.py), JavaScript (.js, .jsx), TypeScript (.ts, .tsx)
    
    Args:
        file_path: Path to the source file.
        
    Returns:
        List of FunctionNode objects with name, line_number, source_code, and docstring.
        Returns empty list if file is empty or has syntax errors.
        
    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    try:
        # Read the file content
        source_code = file_path.read_bytes()
        
        # Handle empty files
        if len(source_code.strip()) == 0:
            logger.info(f"Empty file: {file_path}")
            return []
        
        # Select language based on file extension
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            language = PY_LANGUAGE
            is_python = True
        elif suffix in (".js", ".jsx"):
            language = JS_LANGUAGE
            is_python = False
        elif suffix == ".ts":
            language = TS_LANGUAGE
            is_python = False
        elif suffix == ".tsx":
            language = TSX_LANGUAGE
            is_python = False
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
        
        # Parse with tree-sitter
        parser = Parser(language)
        tree = parser.parse(source_code)
        
        # Check for syntax errors
        if tree.root_node.has_error:
            logger.warning(f"Syntax errors in file: {file_path}")
            return []
        
        # Extract functions
        functions = []
        if is_python:
            _extract_functions(tree.root_node, source_code, file_path, functions)
        else:
            _extract_js_functions(tree.root_node, source_code, file_path, functions)
        
        return functions
        
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return []


def _extract_functions(
    node,
    source_code: bytes,
    file_path: Path,
    functions: list[FunctionNode],
    module_path: str = ""
) -> None:
    """
    Recursively extract function definitions from the AST.
    
    Args:
        node: Current tree-sitter node.
        source_code: The original source code as bytes.
        file_path: Path to the source file.
        functions: List to append FunctionNode objects to.
        module_path: Qualified module path (e.g., "auth.oauth").
    """
    if node.type == "function_definition":
        # Extract function name
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        
        func_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")
        
        # Build qualified name
        if module_path:
            qualified_name = f"{module_path}.{func_name}"
        else:
            # Use file path to build qualified name
            parts = file_path.with_suffix("").parts
            # Find the relevant path (skip leading directories)
            # For now, use filename.function_name
            module_name = file_path.stem
            qualified_name = f"{module_name}.{func_name}"
        
        # Extract line numbers (1-indexed for humans)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        
        # Extract full source code
        func_source = source_code[node.start_byte:node.end_byte].decode("utf-8")
        
        # Extract docstring
        docstring = _extract_docstring(node, source_code)
        
        # Extract function calls
        calls = _extract_function_calls(node, source_code)
        
        # Extract imports (at module level, not function level for now)
        imports: list[str] = []
        
        # Create FunctionNode
        func_node = FunctionNode(
            name=func_name,
            qualified_name=qualified_name,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            source_code=func_source,
            docstring=docstring,
            calls=calls,
            imports=imports
        )
        
        functions.append(func_node)
    
    # Recursively process child nodes
    for child in node.children:
        _extract_functions(child, source_code, file_path, functions, module_path)


def _extract_docstring(node, source_code: bytes) -> str | None:
    """Extract docstring from a function node."""
    body = node.child_by_field_name("body")
    if body is None:
        return None
    
    # Look for the first expression_statement containing a string
    for child in body.children:
        if child.type == "expression_statement":
            for expr_child in child.children:
                if expr_child.type == "string":
                    docstring = source_code[expr_child.start_byte:expr_child.end_byte].decode("utf-8")
                    # Remove quotes
                    return docstring.strip('"""').strip("'''").strip('"').strip("'").strip()
    
    return None


def _extract_function_calls(node, source_code: bytes) -> list[str]:
    """Extract all function calls made within a function."""
    calls = []
    
    def _find_calls(n):
        if n.type == "call":
            func_node = n.child_by_field_name("function")
            if func_node:
                call_name = source_code[func_node.start_byte:func_node.end_byte].decode("utf-8")
                calls.append(call_name)
        
        for child in n.children:
            _find_calls(child)
    
    _find_calls(node)
    return calls


def _extract_js_functions(
    node,
    source_code: bytes,
    file_path: Path,
    functions: list[FunctionNode],
    module_path: str = ""
) -> None:
    """
    Recursively extract function definitions from JavaScript/TypeScript AST.
    
    Extracts:
    - function declarations
    - arrow functions assigned to variables/consts
    - class methods
    - export functions
    
    Args:
        node: Current tree-sitter node.
        source_code: The original source code as bytes.
        file_path: Path to the source file.
        functions: List to append FunctionNode objects to.
        module_path: Qualified module path.
    """
    # Function declaration: function myFunc() {}
    if node.type == "function_declaration":
        _extract_js_function_node(node, source_code, file_path, functions, module_path)
    
    # Arrow function: const myFunc = () => {} or React components
    elif node.type == "lexical_declaration" or node.type == "variable_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                # Check if the value is an arrow function or JSX element (React component)
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                
                if name_node and value_node:
                    # Arrow function or function expression
                    if value_node.type in ("arrow_function", "function"):
                        _extract_js_arrow_function(name_node, value_node, source_code, file_path, functions, module_path)
                    # React component returning JSX
                    elif value_node.type == "arrow_function" or (
                        value_node.type == "parenthesized_expression" and 
                        any(c.type == "jsx_element" or c.type == "jsx_self_closing_element" for c in value_node.children)
                    ):
                        _extract_js_arrow_function(name_node, value_node, source_code, file_path, functions, module_path)
    
    # Method definition in a class
    elif node.type == "method_definition":
        _extract_js_method(node, source_code, file_path, functions, module_path)
    
    # Export declarations - process children but don't recurse again
    elif node.type == "export_statement":
        # Process the exported declaration directly
        for child in node.children:
            if child.type in ("function_declaration", "lexical_declaration", "variable_declaration"):
                _extract_js_functions(child, source_code, file_path, functions, module_path)
        return  # Don't recurse into children again
    
    # Recursively process child nodes
    for child in node.children:
        _extract_js_functions(child, source_code, file_path, functions, module_path)


def _extract_js_function_node(node, source_code: bytes, file_path: Path, functions: list[FunctionNode], module_path: str) -> None:
    """Extract a function declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    
    func_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")
    
    # Build qualified name
    module_name = file_path.stem
    qualified_name = f"{module_name}.{func_name}"
    
    # Extract line numbers (1-indexed)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    
    # Extract full source code
    func_source = source_code[node.start_byte:node.end_byte].decode("utf-8")
    
    # Extract JSDoc comment (JavaScript's equivalent of docstring)
    docstring = _extract_jsdoc(node, source_code)
    
    # Extract function calls
    calls = _extract_js_function_calls(node, source_code)
    
    # Extract imports (at module level, not function level for now)
    imports: list[str] = []
    
    func_node = FunctionNode(
        name=func_name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source_code=func_source,
        docstring=docstring,
        calls=calls,
        imports=imports
    )
    
    functions.append(func_node)


def _extract_js_arrow_function(name_node, value_node, source_code: bytes, file_path: Path, functions: list[FunctionNode], module_path: str) -> None:
    """Extract an arrow function assigned to a variable."""
    func_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")
    
    # Build qualified name
    module_name = file_path.stem
    qualified_name = f"{module_name}.{func_name}"
    
    # Extract line numbers (1-indexed)
    line_start = value_node.start_point[0] + 1
    line_end = value_node.end_point[0] + 1
    
    # Extract full source code (include the const declaration)
    # Go up to the parent to get "const myFunc = ..."
    parent = value_node.parent
    if parent and parent.parent:
        func_source = source_code[parent.parent.start_byte:parent.parent.end_byte].decode("utf-8")
    else:
        func_source = source_code[value_node.start_byte:value_node.end_byte].decode("utf-8")
    
    # Extract JSDoc comment
    docstring = _extract_jsdoc(value_node, source_code)
    
    # Extract function calls
    calls = _extract_js_function_calls(value_node, source_code)
    
    imports: list[str] = []
    
    func_node = FunctionNode(
        name=func_name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source_code=func_source,
        docstring=docstring,
        calls=calls,
        imports=imports
    )
    
    functions.append(func_node)


def _extract_js_method(node, source_code: bytes, file_path: Path, functions: list[FunctionNode], module_path: str) -> None:
    """Extract a class method."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    
    method_name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8")
    
    # Build qualified name (include class name if possible)
    module_name = file_path.stem
    qualified_name = f"{module_name}.{method_name}"
    
    # Extract line numbers (1-indexed)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    
    # Extract full source code
    func_source = source_code[node.start_byte:node.end_byte].decode("utf-8")
    
    # Extract JSDoc comment
    docstring = _extract_jsdoc(node, source_code)
    
    # Extract function calls
    calls = _extract_js_function_calls(node, source_code)
    
    imports: list[str] = []
    
    func_node = FunctionNode(
        name=method_name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source_code=func_source,
        docstring=docstring,
        calls=calls,
        imports=imports
    )
    
    functions.append(func_node)


def _extract_jsdoc(node, source_code: bytes) -> str | None:
    """Extract JSDoc comment from a function node."""
    # Look for comment in the previous siblings or parent
    # JSDoc is typically /** ... */ before the function
    
    # Try to find a comment node before this node
    parent = node.parent
    if parent:
        # Get the index of current node
        try:
            node_index = parent.children.index(node)
            # Look at previous siblings
            for i in range(node_index - 1, -1, -1):
                prev = parent.children[i]
                if prev.type == "comment":
                    comment_text = source_code[prev.start_byte:prev.end_byte].decode("utf-8")
                    # Check if it's a JSDoc comment (starts with /**)
                    if comment_text.strip().startswith("/**"):
                        # Clean up the comment
                        cleaned = comment_text.strip()
                        # Remove /** and */
                        cleaned = cleaned[3:-2].strip()
                        # Remove leading * from each line
                        lines = [line.strip().lstrip("* ").strip() for line in cleaned.split("\n")]
                        return "\n".join(lines).strip()
                # Stop if we hit something other than whitespace/comment
                elif prev.type not in ("comment",):
                    break
        except ValueError:
            pass
    
    return None


def _extract_js_function_calls(node, source_code: bytes) -> list[str]:
    """Extract all function calls made within a JavaScript/TypeScript function."""
    calls = []
    
    def _find_calls(n):
        # Call expression: functionName()
        if n.type == "call_expression":
            func_node = n.child_by_field_name("function")
            if func_node:
                call_name = source_code[func_node.start_byte:func_node.end_byte].decode("utf-8")
                calls.append(call_name)
        
        for child in n.children:
            _find_calls(child)
    
    _find_calls(node)
    return calls
