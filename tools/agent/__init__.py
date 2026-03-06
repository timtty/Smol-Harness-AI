"""
Agent-specific tools (e.g. task queue). Stateful; shared across agent runs in same process.
"""

from typing import List
from langchain.tools import tool

_pending_tasks: List[dict] = []


@tool(description="Queue a new task to be executed. Use this when you discover new work during execution, such as finding URLs that each need to be read separately.")
def add_task(description: str, action: str, input_hint: str) -> str:
    _pending_tasks.append({
        "description": description,
        "action": action,
        "input_hint": input_hint,
    })
    return f"Task queued: {description}"


def get_pending_tasks() -> List[dict]:
    return _pending_tasks


def clear_pending_tasks() -> None:
    _pending_tasks.clear()


__all__ = ["add_task", "get_pending_tasks", "clear_pending_tasks"]
