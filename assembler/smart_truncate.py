"""Smart WARM tier compression without LLM calls."""

from models import FunctionNode


def smart_truncate(func: FunctionNode, max_lines: int = 6) -> str:
    """
    Extract most meaningful lines using structural patterns.

    This avoids LLM calls for WARM tier compression while maintaining
    code understanding through AST-aware heuristics.

    Args:
        func: FunctionNode containing source code and metadata.
        max_lines: Maximum number of lines to extract (default 6).

    Returns:
        Formatted string with signature + docstring + key logic + return.
    """
    lines = func.source_code.split("\n")
    important = []

    # 1. Signature — find first non-comment line
    signature = None
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            signature = line
            break

    if signature:
        important.append(signature)

    # 2. Docstring with fallback
    if func.docstring:
        first_line = func.docstring.split("\n")[0].strip()
        if len(first_line) > 10:
            important.append(f'"""{first_line}"""')
    else:
        # Parser missed it — scan source directly
        for line in lines[1:6]:
            stripped = line.strip()
            if stripped.startswith(('"""', "'''")):
                important.append(stripped[:80])
                break

    # 3. Meaningful lines — skip signature to avoid duplication
    start_idx = 1 if signature else 0
    meaningful = [
        l for l in lines[start_idx:]
        if l.strip() and not l.strip().startswith(("#", "def ", "class ", '"', "'"))
    ]
    important.extend(meaningful[:3])

    # 4. Return statement — last actual return, not comments/variables
    # Use reversed to get the last return statement
    for line in reversed(lines):
        stripped = line.strip()
        # Only match actual return keywords, not comments or variable names
        if stripped.startswith("return ") or stripped == "return":
            if len(important) < max_lines:  # Budget check
                important.append(line)
            break

    return "\n".join(important)


def smart_truncate_batch(functions: list[FunctionNode], max_lines_per_func: int = 6) -> dict[str, str]:
    """
    Smart truncate multiple functions without LLM compression.

    Args:
        functions: List of FunctionNode objects to compress.
        max_lines_per_func: Maximum lines per function (default 6).

    Returns:
        Dict mapping qualified_name → truncated content.
    """
    return {
        func.qualified_name: smart_truncate(func, max_lines_per_func)
        for func in functions
    }