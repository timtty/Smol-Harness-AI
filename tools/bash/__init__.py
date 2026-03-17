"""
Bash/shell execution tool.
"""

from langchain.tools import tool
from subprocess import run as subprocess_run
from subprocess import PIPE
from subprocess import TimeoutExpired

# Max chars returned so the model stays within context; when exceeded, output is truncated with a note
BASH_OUTPUT_MAX_CHARS = 100_000
TRUNCATE_SUFFIX = "\n\n[... output truncated to fit context ...]"


@tool(description="Execute a bash/shell command and return stdout, stderr, and exit code. Use for: AWS CLI (e.g. aws s3 ls, aws s3api list-buckets), gcloud, other CLIs, running scripts, listing files, and any task that explicitly asks to 'use bash' or 'run commands'. Avoid long-running or interactive commands.")
def bash_execute(command: str, timeout_seconds: int = 30) -> str:
    try:
        proc = subprocess_run(
            command,
            shell=True,  # nosec B602 — intentional: this tool exists to execute arbitrary shell commands
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        output = proc.stdout.strip()
        errors = proc.stderr.strip()
        parts = [f"exit_code: {proc.returncode}"]
        if output:
            parts.append(f"stdout:\n{output}")
        if errors:
            parts.append(f"stderr:\n{errors}")
        result = "\n".join(parts)
        if len(result) > BASH_OUTPUT_MAX_CHARS:
            result = result[: BASH_OUTPUT_MAX_CHARS] + TRUNCATE_SUFFIX
        return result
    except TimeoutExpired:
        return f"Command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"Error executing command: {e}"


__all__ = ["bash_execute"]
