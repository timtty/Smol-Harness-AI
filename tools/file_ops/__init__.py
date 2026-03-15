"""
File operations: read, write, and targeted line search for local files.
"""

from langchain.tools import tool


@tool(description="Read the contents of a local file by path")
def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool(description="Write content to a local file by path")
def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File written: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(description=(
    "Search a file for lines containing a substring and return only the matching lines "
    "with surrounding context and line numbers. Use this before editing a file to locate "
    "the exact lines to change without loading the entire file into context. "
    "Returns compact blocks: '→' marks matched lines, plain numbers are context. "
    "context_lines controls how many lines before/after each match to include (default 3)."
))
def find_lines(path: str, pattern: str, context_lines: int = 3) -> str:
    """Find lines in a file matching a substring pattern, returning minimal context blocks."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    needle = pattern.lower()
    matched = [i for i, line in enumerate(lines) if needle in line.lower()]

    if not matched:
        return f"No matches for '{pattern}' in {path}"

    # Merge match indices into contiguous blocks with context
    blocks: list[tuple[int, int]] = []
    start = max(0, matched[0] - context_lines)
    end = min(len(lines) - 1, matched[0] + context_lines)
    for idx in matched[1:]:
        blk_start = max(0, idx - context_lines)
        blk_end = min(len(lines) - 1, idx + context_lines)
        if blk_start <= end + 1:
            end = max(end, blk_end)
        else:
            blocks.append((start, end))
            start, end = blk_start, blk_end
    blocks.append((start, end))

    matched_set = set(matched)
    parts = []
    for blk_start, blk_end in blocks:
        chunk = []
        for i in range(blk_start, blk_end + 1):
            prefix = "→" if i in matched_set else " "
            chunk.append(f"{prefix} {i + 1:>4}: {lines[i].rstrip()}")
        parts.append("\n".join(chunk))

    total = len(matched)
    header = f"{total} match{'es' if total != 1 else ''} for '{pattern}' in {path}:\n"
    return header + "\n---\n".join(parts)


@tool(description=(
    "Edit a local file by replacing an exact string with new content. "
    "Use find_lines first to locate the exact text to replace. "
    "The old_string must match exactly (including whitespace and indentation). "
    "Returns a confirmation message or an error if old_string is not found."
))
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in the file at path."""
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    if old_string not in content:
        return f"old_string not found in {path}"

    new_content = content.replace(old_string, new_string, 1)
    try:
        with open(path, "w") as f:
            f.write(new_content)
        return f"File edited: {path}"
    except PermissionError:
        return f"Permission denied writing: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


file_ops_tooling = [
    read_file,
    write_file,
    edit_file,
    find_lines,
]

__all__ = ["read_file", "write_file", "edit_file", "find_lines", "file_ops_tooling"]
