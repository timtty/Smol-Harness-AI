---
name: agent-coding-standard
version: 1.0.0
description: Coding style and architectural patterns for LangGraph-based agents in this project. Read this before writing or extending any agent file.
author: tremcom
tags:
  - langgraph
  - python
  - agent
  - architecture
  - coding-standard
type: spec
applies_to: "*.py"
---

# Agent Coding Standard

This document defines the coding style and architectural patterns used in this project. All agents and contributors must follow these conventions when building or extending agent files.

Reference implementation: `basic_agent.py`

---

## Framework Stack

| Layer | Library |
|---|---|
| Graph orchestration | `langgraph` (`StateGraph`) |
| LLM interface | `langchain` + `langchain_openai` (`ChatOpenAI`) |
| Console output | `rich` |
| Environment | `python-dotenv` |
| State types | `typing_extensions` (`TypedDict`, `List`, `Annotated`) |

---

## Import Style

- One import per line. No multi-import lines.
- Alias verbose names at import time for readability:

```python
from json import dumps as json_to_string
from json import loads as json_parse
from os import environ as env
from random import choice as random_choice
```

- Group imports in this order:
  1. LangGraph / LangChain
  2. Typing
  3. Standard library
  4. Third-party (rich, dotenv, etc.)
  5. Local packages (`tools`, etc.)

---

## LLM Instantiation

Never instantiate the LLM at module level. Always use a factory function:

```python
def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        base_url=env.get("OPENAI_BASE_URL"),
        api_key=env.get("OPENAI_API_KEY"),
    )
```

- `MODEL` is a module-level constant string.
- Credentials always come from environment variables via `env.get(...)`.
- Use `llm.with_structured_output(Schema, include_raw=True)` for structured outputs so token usage is accessible via `result["raw"].usage_metadata`.

---

## State Design

All state types are `TypedDict`. Define separate types for graph input, working state, and output:

```python
class StartState(TypedDict):
    start_prompt: str

class AgentState(TypedDict):
    run_id: str
    prompt: str
    tasks: List[Task]
    current_task_id: int
    iterations: int
    termination_reason: str

class EndState(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_sec: int
    execution_time: int
```

Structured output schemas (for LLM responses) are also `TypedDict`:

```python
class ValidationResult(TypedDict):
    valid: bool
    reason: str
```

Pass `input_schema` and `output_schema` to `StateGraph`:

```python
builder = StateGraph(AgentState, input_schema=StartState, output_schema=EndState)
```

---

## Graph Architecture (Plan-Execute-Validate Loop)

The standard agent graph follows this node sequence:

```
START -> agent_bootstrap -> agent_plan -> task_manager -> agent_execute -> agent_validate
                                               ^                                 |
                                               |_________________________________|
                                               (retry or next task)
                                         task_manager -> agent_terminate -> END
```

**Required nodes:**

| Node | Responsibility |
|---|---|
| `agent_bootstrap` | Initialize `run_id`, state fields, start console spinner |
| `planning_agent` | Decompose user prompt into an ordered `Task` list |
| `task_manager` | Pick next pending task; enforce `MAX_ITERATIONS`; set termination reason |
| `agent_execute` | Run tools or LLM to fulfill the current task |
| `agent_validate` | LLM-based pass/fail check; retry up to 3 times; store validated results |
| `agent_terminate` | Synthesize final response from store; emit token/time stats |

**Routing functions** determine conditional edges:

```python
def route_task_manager(state: AgentState) -> str:
    return "agent_terminate" if state["termination_reason"] else "agent_execute"

def route_validate(state: AgentState) -> str:
    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])
    return "agent_execute" if current["status"] == "in_progress" else "task_manager"
```

---

## Task Schema

Tasks are `TypedDict` with this exact shape:

```python
class Task(TypedDict):
    task_id: int
    description: str
    action: str       # website_search | read_webpage | fetch_rss_articles | read_file | write_file | bash_execute | summarize
    input_hint: str
    status: str       # pending | in_progress | complete | failed
    result: str
    retry_count: int
    error_context: List[str]
```

Tasks are **never mutated in place**. Use list comprehension to produce updated lists:

```python
updated_tasks = [
    {**t, "status": "complete"} if t["task_id"] == target_id else t
    for t in state["tasks"]
]
```

---

## Memory / Store

Use `InMemoryStore` compiled into the graph. Provide a safe accessor that falls back gracefully:

```python
memory_store = InMemoryStore()

def get_store() -> BaseStore:
    try:
        store = _get_store()
        return store if store is not None else memory_store
    except (RuntimeError, KeyError):
        return memory_store
```

Store validated task results under the namespace `("task_results", run_id)` keyed by `task_id`.

Use `_bm25_search()` for relevance-ranked retrieval from the store rather than raw `store.search()` when the result count may be large.

---

## LLM Caching

Enable SQLite LLM caching by default. Allow bypass via env var or CLI flag:

```python
if not env.get("FRESH_RUN") and "--no-cache" not in sys.argv:
    set_llm_cache(SQLiteCache(database_path=".cache/llm.db"))
```

Cache files go in `.cache/`. Create the directory at startup:

```python
_os.makedirs(".cache", exist_ok=True)
```

---

## Console / Logging

Use `rich.console.Console` for all output. Never use `print()`.

```python
console = Console()
status = console.status("Waking up...", spinner="aesthetic")
```

The `log()` function prints with a random pastel color bullet:

```python
PASTEL_COLORS = ["#FFB3BA", "#FFDFBA", ...]

def log(message: str) -> None:
    color = random_choice(PASTEL_COLORS)
    console.print(f"[{color}]●[/] {message}\n")
```

Use `rich.tree.Tree`, `rich.table.Table`, and `rich.markdown.Markdown` for structured tool output display.

Spinner states:
- `"aesthetic"` — during LLM thinking
- `"bouncingBall"` — while agent is running between LLM calls

---

## Spinner / Status Management

Use `@contextmanager` for the thinking ticker so it is exception-safe:

```python
@contextmanager
def _thinking_with_elapsed_ticker(status_obj, min_seconds_before_ticker=3, tick_interval=1.0):
    ...
    yield
    ...
```

Always stop the background ticker before invoking the LLM, and restart it after:

```python
_stop_agent_running_ticker()
status.update(status="Thinking...", spinner="aesthetic")
with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
    result = llm.invoke(...)
status.update(status="Agent is running...", spinner="bouncingBall")
_start_agent_running_ticker(status)
```

---

## Token Tracking

Track cumulative token usage in module-level globals:

```python
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0
```

After every LLM call, increment both:

```python
INPUT_TOKENS += result["raw"].usage_metadata["input_tokens"]
OUTPUT_TOKENS += result["raw"].usage_metadata["output_tokens"]
```

Report totals in `agent_terminate`.

---

## Truncation Helpers

Define truncation helpers to avoid blowing context on validation and summarize steps:

```python
VALIDATE_RESULT_MAX_CHARS = 6_000
SUMMARIZE_RESULT_MAX_CHARS = 20_000

def _truncate_for_validation(text: str, max_chars: int = VALIDATE_RESULT_MAX_CHARS) -> str: ...
def _truncate_for_summarize(text: str, max_chars: int = SUMMARIZE_RESULT_MAX_CHARS) -> str: ...
```

Always use these helpers when injecting task results into prompts. Never pass raw, unbounded result strings to the LLM.

---

## Bash Safety

All `bash_execute` calls must prompt for user confirmation before execution:

```python
approved = Confirm.ask(f"[yellow]Allow bash:[/] [bold]{tool_args.get('command', '')}[/]")
if not approved:
    result = "User declined to execute this command."
    continue
```

Never run bash commands silently.

---

## Tools

Tools are defined in the `tools/` package and imported as:

```python
from tools import all_tools, tool_map
```

- `all_tools` — list passed to `llm.bind_tools()`
- `tool_map` — dict for manual dispatch by name

Dynamic tasks added mid-run come through `tools.agent.get_pending_tasks()` / `clear_pending_tasks()`. Check for these after each `agent_execute` step and append them to the task list.

---

## Iteration Safety

Always enforce a hard cap on graph cycles:

```python
MAX_ITERATIONS = 20
```

Check in `task_manager`:

```python
if state["iterations"] >= MAX_ITERATIONS:
    return {"termination_reason": "max_iterations"}
```

Termination reasons: `"complete"`, `"impossible"`, `"max_iterations"`.

---

## Retry Logic

- Max retries per task: **3**
- Track retry state via `retry_count` and `error_context` on the `Task`
- On validation failure: append the validator's reason to `error_context`, increment `retry_count`, keep status `"in_progress"` to re-enter `agent_execute`
- On 3rd failure: set status to `"failed"` and let `task_manager` advance

---

## Graph Compilation

Compile with the shared memory store:

```python
agent = builder.compile(store=memory_store)
```

---

## CLI Entry Point

Use `argparse` with an optional positional `prompt` argument and `--no-cache` flag. Default prompt should exercise all major agent capabilities.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
```
