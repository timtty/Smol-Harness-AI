"""
File operations: read and write local files.
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


file_ops_tooling = [
    read_file,
    write_file,
]

__all__ = ["read_file", "write_file", "file_ops_tooling"]
