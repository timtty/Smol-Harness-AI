from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

from langchain.messages import SystemMessage
from langchain.messages import HumanMessage

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.config import get_store as _get_store
from typing_extensions import TypedDict
from typing_extensions import List
from typing_extensions import Annotated
from shortuuid import uuid as suid

from json import dumps as json_to_string
from json import loads as json_parse

import sys
from dotenv import load_dotenv
from os import environ as env
from random import choice as random_choice
from time import time
from time import sleep
from contextlib import contextmanager
import threading

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich.table import Table
from rich.tree import Tree
from rich import box



console = Console()
console.clear()
status = console.status("Waking up...", spinner="aesthetic")

load_dotenv()

import os as _os
_os.makedirs(".cache", exist_ok=True)
from langchain_community.cache import SQLiteCache
from langchain_classic.globals import set_llm_cache

# Skip LLM cache when FRESH_RUN=1 or --no-cache so the same prompt runs without cached responses
if not env.get("FRESH_RUN") and "--no-cache" not in sys.argv:
    set_llm_cache(SQLiteCache(database_path=".cache/llm.db"))

PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#E8BAFF", "#FFB3E6", "#B3F0FF", "#C9FFE5", "#FFD9B3",
]

def log(message: str) -> None:
    color = random_choice(PASTEL_COLORS)
    console.print(f"[{color}]●[/] {message}\n")

MODEL = "qwen/qwen3-4b-2507"

# Max chars of task result to send to validator (must fit in model context with system + task; keep conservative for small context models)
VALIDATE_RESULT_MAX_CHARS = 6_000
# Max chars per result when building context for summarize step (5 results × this = total cap)
SUMMARIZE_RESULT_MAX_CHARS = 20_000


def _truncate_for_validation(text: str, max_chars: int = VALIDATE_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... result truncated for validation ...]"


def _truncate_for_summarize(text: str, max_chars: int = SUMMARIZE_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for context ...]"


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        base_url=env.get("OPENAI_BASE_URL"),
        api_key=env.get("OPENAI_API_KEY"),
    )


memory_store = InMemoryStore()


def get_store() -> BaseStore:
    """Use runtime store when available, else the graph's memory_store (same instance passed to compile)."""
    try:
        store = _get_store()
        return store if store is not None else memory_store
    except (RuntimeError, KeyError):
        return memory_store


def _bm25_search(store: BaseStore, namespace: tuple, query: str, limit: int) -> list:
    all_items = store.search(namespace, limit=1000)
    if not all_items or len(all_items) <= limit:
        return all_items
    docs = [
        Document(
            page_content=f"{item.value['description']} {item.value['result']}",
            metadata={"key": item.key},
        )
        for item in all_items
    ]
    retriever = BM25Retriever.from_documents(docs, k=limit)
    ranked = retriever.invoke(query)
    ranked_keys = {doc.metadata["key"] for doc in ranked}
    return [item for item in all_items if item.key in ranked_keys]

MAX_ITERATIONS = 20
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0
START_TIME = time()
CONTEXT_WINDOW_MAX = 100_000


def _ctx_status() -> str:
    used = INPUT_TOKENS + OUTPUT_TOKENS
    pct = int(used / CONTEXT_WINDOW_MAX * 100)
    return f"Agent is running... (context window: {used // 1000}k/{CONTEXT_WINDOW_MAX // 1000}k - {pct}%)"


def _token_status() -> str:
    """Status line during agent work (no token counts)."""
    return "Agent is running..."


_agent_running_ticker_stop: threading.Event | None = None
_agent_running_ticker_interval = 0.5


def _start_agent_running_ticker(status_obj) -> None:
    """Start background ticker that updates status during agent work."""
    global _agent_running_ticker_stop
    if _agent_running_ticker_stop is not None:
        _agent_running_ticker_stop.set()
    _agent_running_ticker_stop = threading.Event()

    def ticker():
        while not _agent_running_ticker_stop.wait(_agent_running_ticker_interval):
            status_obj.update(status=_token_status(), spinner="bouncingBall")

    t = threading.Thread(target=ticker, daemon=True)
    t.start()


def _stop_agent_running_ticker() -> None:
    """Stop the agent-running token ticker."""
    global _agent_running_ticker_stop
    if _agent_running_ticker_stop is not None:
        _agent_running_ticker_stop.set()
        _agent_running_ticker_stop = None


@contextmanager
def _thinking_with_elapsed_ticker(status_obj, min_seconds_before_ticker: int = 3, tick_interval: float = 1.0):
    """Show 'Thinking...' during LLM invoke, then after min_seconds_before_ticker show 'Thinking... Ns', updating every tick_interval."""
    stop_ev = threading.Event()
    start_time = time()

    def ticker():
        if stop_ev.wait(min_seconds_before_ticker):
            return
        while not stop_ev.is_set():
            elapsed = int(time() - start_time)
            status_obj.update(status=f"Thinking... {elapsed}s", spinner="aesthetic")
            if stop_ev.wait(tick_interval):
                break

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_ev.set()
        t.join(timeout=1.0)


# ── Tools (from tools package) ───────────────────────────────────────────────────

from tools import all_tools, tool_map
from tools.agent import get_pending_tasks, clear_pending_tasks
from tools.bash import TRUNCATE_SUFFIX as BASH_TRUNCATE_SUFFIX


# ── State ──────────────────────────────────────────────────────────────────────

class StartState(TypedDict):
    start_prompt: str


class EndState(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_sec: int
    execution_time: int


class Task(TypedDict):
    task_id: int
    description: str
    action: str            # website_search | read_webpage | fetch_rss_articles | read_file | write_file | bash_execute | summarize
    input_hint: str
    status: str            # pending | in_progress | complete | failed
    result: str
    retry_count: int
    error_context: List[str]


class AgentState(TypedDict):
    run_id: str
    prompt: str
    tasks: List[Task]
    current_task_id: int
    iterations: int
    termination_reason: str


# ── Planner structured output schemas ─────────────────────────────────────────

class PlanStep(TypedDict):
    step_id: int
    description: str
    action: str      # website_search | read_webpage | fetch_rss_articles | read_file | write_file | bash_execute | summarize
    input_hint: str  # specific query, URL, path, or instruction for this step

class ExecutionPlan(TypedDict):
    steps: List[PlanStep]


# ── Validator structured output schema ────────────────────────────────────────

class ValidationResult(TypedDict):
    valid: bool
    reason: str


# ── Task list display ──────────────────────────────────────────────────────────

def print_task_list(tasks: List[Task]) -> None:
    status_icon = {
        "complete":    ("[bold green]✔[/]",   lambda t: f"[strike dim]{t}[/]"),
        "failed":      ("[bold red]✘[/]",     lambda t: f"[strike dim]{t}[/]"),
        "in_progress": ("[bold yellow]▶[/]",  lambda t: t),
        "pending":     ("[dim]○[/]",          lambda t: t),
    }
    console.print()
    console.print("[bold]Task List[/bold]")
    console.print()
    for i, task in enumerate(tasks, 1):
        icon, fmt = status_icon.get(task["status"], status_icon["pending"])
        console.print(f"  {i}. {icon}  {fmt(task['description'])}")
    done = sum(1 for t in tasks if t["status"] == "complete")
    console.print()
    console.print(f"  [dim]{done}/{len(tasks)} tasks complete[/dim]\n")


# ── Nodes ──────────────────────────────────────────────────────────────────────

def agent_bootstrap(state: StartState) -> AgentState:
    status.start()
    log("Bootstrapping agent...")
    START_TIME = time()
    return {
        "run_id": suid(),
        "prompt": state["start_prompt"],
        "tasks": [],
        "current_task_id": 0,
        "iterations": 0,
        "termination_reason": "",
    }


def planning_agent(state: AgentState) -> AgentState:
    llm = _llm()
    planner = llm.with_structured_output(ExecutionPlan, include_raw=True)

    guidance_prompt = SystemMessage(content="""You are a planning agent. Decompose the user's task into a minimal, ordered sequence of executable steps.

Available actions:
- website_search    — query the web for information
- read_webpage      — fetch and read a URL (HTML pages, articles)
- fetch_rss_articles — fetch an RSS/Atom feed and return a compact list of articles; use this instead of read_webpage when the URL is an RSS feed (.xml)
- read_file         — read a local file by path
- write_file        — write content to a local file
- bash_execute      — run bash/shell commands (e.g. AWS CLI: aws s3 ls, aws s3api list-buckets; gcloud; scripts). Use when the user asks to use AWS CLI, list buckets, run commands via bash, or any CLI/shell task.
- summarize         — synthesize gathered content into a final output

For each step output:
  step_id     : sequential integer starting at 1
  description : what this step accomplishes
  action      : one of the actions above
  input_hint  : the specific query, URL, path, or instruction for this step

URL usage rules:
- Use website_search ONLY when a URL is not yet known.
- If a prior step will produce URLs (e.g. a search or feed list), plan subsequent steps as read_webpage and set input_hint to describe which URL to extract from the prior step result (e.g. "use the first article URL from step 1 result").
- Never search for something that can be read directly from a URL obtained in a prior step.

Be concise. Do not over-plan. Only include steps that are necessary.""")

    action_prompt = HumanMessage(content=state["prompt"])

    global INPUT_TOKENS, OUTPUT_TOKENS
    log("Planning...")
    _stop_agent_running_ticker()
    status.update(status="Thinking...", spinner="aesthetic")
    with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
        result = planner.invoke([guidance_prompt, action_prompt])
    status.update(status="Agent is running...", spinner="bouncingBall")
    _start_agent_running_ticker(status)

    INPUT_TOKENS += result["raw"].usage_metadata["input_tokens"]
    OUTPUT_TOKENS += result["raw"].usage_metadata["output_tokens"]

    tasks = [
        {
            "task_id": step["step_id"],
            "description": step["description"],
            "action": step["action"],
            "input_hint": step["input_hint"],
            "status": "pending",
            "result": "",
            "retry_count": 0,
            "error_context": [],
        }
        for step in result["parsed"]["steps"]
    ]
    log(f"Created a plan with {len(result['parsed']['steps'])} task(s)")
    print_task_list(tasks)
    return {"tasks": tasks}


def task_manager(state: AgentState) -> AgentState:
    if state["iterations"] >= MAX_ITERATIONS:
        log(f"Max iterations ({MAX_ITERATIONS}) reached, terminating...")
        return {"termination_reason": "max_iterations"}

    pending = [t for t in state["tasks"] if t["status"] == "pending"]

    if not pending:
        all_failed = all(t["status"] == "failed" for t in state["tasks"])
        reason = "impossible" if all_failed else "complete"
        log(f"No pending tasks, task list {reason}, terminating...")
        return {"termination_reason": reason}

    next_task = pending[0]
    log(f"Queuing task {next_task['task_id']}/{len(state['tasks'])}: {next_task['description']}")
    updated_tasks = [
        {**t, "status": "in_progress"} if t["task_id"] == next_task["task_id"] else t
        for t in state["tasks"]
    ]
    print_task_list(updated_tasks)
    return {
        "tasks": updated_tasks,
        "current_task_id": next_task["task_id"],
        "iterations": state["iterations"] + 1,
        "termination_reason": "",
    }


def agent_execute(state: AgentState) -> AgentState:
    global INPUT_TOKENS, OUTPUT_TOKENS
    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])

    retry_suffix = f" (retry {current['retry_count']}/3)" if current["retry_count"] > 0 else ""
    log(f"Executing task {current['task_id']}/{len(state['tasks'])}: {current['description']}{retry_suffix}")

    retry_context = ""
    if current["error_context"]:
        retry_context = "\n\nPrevious attempt errors:\n" + "\n".join(current["error_context"])

    if current["action"] == "summarize":
        llm = _llm()
        store = get_store()
        hits = _bm25_search(store, ("task_results", state["run_id"]), current["description"], 5)
        completed_results = "\n\n".join(
            f"Task {h.key} — {h.value['description']}:\n{_truncate_for_summarize(h.value['result'])}"
            for h in hits
        ) or "No prior results available."
        guidance_prompt = SystemMessage(content="You are a summarization agent. Synthesize the provided information into a clear, concise summary.")
        action_prompt = HumanMessage(content=f"Task: {current['description']}\n\nContext:\n{completed_results}{retry_context}")
        _stop_agent_running_ticker()
        status.update(status="Thinking...", spinner="aesthetic")
        llm_response = llm.invoke([guidance_prompt, action_prompt])
        status.update(status="Agent is running...", spinner="bouncingBall")
        _start_agent_running_ticker(status)
        INPUT_TOKENS += llm_response.usage_metadata["input_tokens"]
        OUTPUT_TOKENS += llm_response.usage_metadata["output_tokens"]
        result = llm_response.content

    else:
        llm = _llm()
        executor = llm.bind_tools(all_tools)
        guidance_prompt = SystemMessage(content="""You are a task execution agent. Use the available tools to complete the task.
Use bash_execute when the task involves running shell commands, AWS CLI (e.g. list buckets, S3, EC2), or any other CLI. For "list buckets" or "use aws cli" use: bash_execute with command e.g. "aws s3 ls" or "aws s3api list-buckets".
If prior task results are provided, extract URLs or data from them rather than performing a new search.
If during execution you discover new work items (e.g. multiple URLs that each need to be read), use add_task to queue them rather than trying to handle everything yourself.
If validation failed because output was truncated: retry with a different approach to get full data. For AWS list-buckets or large listings, use pagination (e.g. aws s3api list-buckets --max-items 20, then --starting-token for the next page; or split into multiple commands). Prefer commands that return complete, untruncated results.""")
        store = get_store()
        hits = _bm25_search(store, ("task_results", state["run_id"]), current["description"], 3)
        prior_context = (
            "\n\nRelevant prior results:\n" +
            "\n\n".join(
                f"Task {h.key} — {h.value['description']}:\n{_truncate_for_summarize(h.value['result'])}"
                for h in hits
            )
            if hits else ""
        )
        action_prompt = HumanMessage(content=f"Task: {current['description']}\nSuggested input: {current['input_hint']}{prior_context}{retry_context}")
        _stop_agent_running_ticker()
        status.update(status="Thinking...", spinner="aesthetic")
        with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
            response = executor.invoke([guidance_prompt, action_prompt])
        status.update(status="Agent is running...", spinner="bouncingBall")
        _start_agent_running_ticker(status)
        INPUT_TOKENS += response.usage_metadata["input_tokens"]
        OUTPUT_TOKENS += response.usage_metadata["output_tokens"]

        if response.tool_calls:
            result = ""
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                first_arg = next(iter(tool_args.values()), "")

                if tool_name == "add_task":
                    tool_map[tool_name].invoke(tool_args)
                    log(f"New task queued: [dim]{tool_args.get('description', '')}[/]")
                    continue

                if tool_name == "bash_execute":
                    status.stop()
                    approved = Confirm.ask(f"[yellow]Allow bash:[/] [bold]{tool_args.get('command', '')}[/]")
                    status.start()
                    if not approved:
                        result = "User declined to execute this command."
                        log("Bash command declined by user")
                        continue

                tool_tree = Tree(f'[bold]● {tool_name}[/]([cyan]"{first_arg}"[/])')
                t0 = time()
                result = str(tool_map[tool_name].invoke(tool_args))
                elapsed = time() - t0

                if tool_name == "website_search":
                    parsed = json_parse(result)
                    tool_tree.add(f"[green]Retrieved {len(parsed)} results[/]")
                    console.print(tool_tree)

                    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
                    table.add_column("#", style="dim", width=3)
                    table.add_column("Title")
                    table.add_column("URL", style="dim cyan")
                    for i, r in enumerate(parsed[:5], 1):
                        table.add_row(str(i), r["title"], r["url"])
                    console.print(table)
                    console.print(f"  [dim]elapsed: {elapsed:.2f}s[/dim]\n")

                elif tool_name == "fetch_rss_articles":
                    parsed = json_parse(result)
                    tool_tree.add(f"[green]Retrieved {len(parsed)} articles[/]")
                    console.print(tool_tree)
                    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
                    table.add_column("#", style="dim", width=3)
                    table.add_column("Title")
                    table.add_column("URL", style="dim cyan")
                    for i, r in enumerate(parsed, 1):
                        table.add_row(str(i), r["title"], r["url"])
                    console.print(table)
                    console.print(f"  [dim]elapsed: {elapsed:.2f}s[/dim]\n")

                elif tool_name == "read_webpage":
                    tool_tree.add(f"[green]Read {len(result):,} chars[/]")
                    console.print(tool_tree)
                    console.print(Markdown(result[:250]))

                elif tool_name == "read_file":
                    tool_tree.add(f"[green]Read {len(result):,} chars[/]")
                    console.print(tool_tree)

                elif tool_name == "write_file":
                    tool_tree.add(f"[green]{result}[/]")
                    console.print(tool_tree)

                elif tool_name == "bash_execute":
                    lines = result.splitlines()
                    exit_line = next((l for l in lines if l.startswith("exit_code:")), "")
                    exit_code = exit_line.split(":", 1)[-1].strip() if exit_line else "?"
                    color = "green" if exit_code == "0" else "red"
                    tool_tree.add(f"[{color}]exit {exit_code}[/] — {len(result):,} chars")
                    console.print(tool_tree)
                    preview = "\n".join(lines[1:6])
                    if preview:
                        console.print(f"  [dim]{preview}[/dim]")

                else:
                    tool_tree.add(f"[green]{result[:120]}[/]")
                    console.print(tool_tree)

        else:
            result = response.content

    log(f"Task {current['task_id']} execution complete")
    updated_tasks = [
        {**t, "result": result} if t["task_id"] == state["current_task_id"] else t
        for t in state["tasks"]
    ]

    pending = get_pending_tasks()
    if pending:
        next_id = max(t["task_id"] for t in updated_tasks) + 1
        new_tasks = [
            {
                "task_id": next_id + i,
                "description": t["description"],
                "action": t["action"],
                "input_hint": t["input_hint"],
                "status": "pending",
                "result": "",
                "retry_count": 0,
                "error_context": [],
            }
            for i, t in enumerate(pending)
        ]
        clear_pending_tasks()
        updated_tasks = updated_tasks + new_tasks
        log(f"Added {len(new_tasks)} new task(s) to the plan")
        print_task_list(updated_tasks)

    return {"tasks": updated_tasks}


def agent_validate(state: AgentState) -> AgentState:
    global INPUT_TOKENS, OUTPUT_TOKENS
    store = get_store()
    llm = _llm()
    validator = llm.with_structured_output(ValidationResult, include_raw=True)

    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])

    log(f"Validating task {current['task_id']}...")

    guidance_prompt = SystemMessage(content="""You are a validation agent. Assess whether the task was completed successfully.
Return valid=true if the result meaningfully addresses the task description and is complete.
Return valid=false with a reason if the result is empty, an error, irrelevant, clearly incomplete, or if the result explicitly states that output was truncated (e.g. "[... output truncated to fit context ...]"). When output was truncated, the reason must ask to retry with a different approach to get full data (e.g. pagination, --max-items, multiple requests).""")

    result_for_prompt = _truncate_for_validation(current["result"])
    action_prompt = HumanMessage(content=f"Task: {current['description']}\nResult: {result_for_prompt}")

    _stop_agent_running_ticker()
    status.update(status="Thinking...", spinner="aesthetic")
    with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
        verdict = validator.invoke([guidance_prompt, action_prompt])
    status.update(status="Agent is running...", spinner="bouncingBall")
    _start_agent_running_ticker(status)
    raw = verdict.get("raw")
    if raw and getattr(raw, "usage_metadata", None):
        INPUT_TOKENS += raw.usage_metadata.get("input_tokens", 0)
        OUTPUT_TOKENS += raw.usage_metadata.get("output_tokens", 0)

    parsed = verdict.get("parsed") or {}
    valid = parsed.get("valid", False)
    reason = parsed.get("reason", "Validation response could not be parsed")

    # If bash (or similar) output was truncated, force failure so the agent retries with pagination/different approach
    if BASH_TRUNCATE_SUFFIX in (current.get("result") or ""):
        valid = False
        reason = "Output was truncated. Retry with a different approach to retrieve full data (e.g. pagination: --max-items, --starting-token; or multiple smaller commands)."

    if valid:
        log(f"Task {current['task_id']} passed validation")
        store.put(
            ("task_results", state["run_id"]),
            str(current["task_id"]),
            {
                "description": current["description"],
                "result": current["result"],
                "action": current["action"],
            }
        )
        updated_tasks = [
            {**t, "status": "complete"} if t["task_id"] == state["current_task_id"] else t
            for t in state["tasks"]
        ]
        print_task_list(updated_tasks)
        return {"tasks": updated_tasks}

    new_retry_count = current["retry_count"] + 1

    if new_retry_count >= 3:
        log(f"Task {current['task_id']} failed validation after 3 attempts, marking failed")
        updated_tasks = [
            {**t, "status": "failed", "retry_count": new_retry_count} if t["task_id"] == state["current_task_id"] else t
            for t in state["tasks"]
        ]
    else:
        log(f"Task {current['task_id']} failed validation ({new_retry_count}/3): {reason}")
        updated_tasks = [
            {**t, "status": "in_progress", "retry_count": new_retry_count, "error_context": current["error_context"] + [reason]}
            if t["task_id"] == state["current_task_id"] else t
            for t in state["tasks"]
        ]

    print_task_list(updated_tasks)
    return {"tasks": updated_tasks}


def agent_terminate(state: AgentState) -> EndState:
    llm = _llm()
    store = get_store()

    failed = [t for t in state["tasks"] if t["status"] == "failed"]

    log(f"Terminating ({state['termination_reason']}): {len(failed)} failed. Generating final response...")

    hits = _bm25_search(store, ("task_results", state["run_id"]), state["prompt"], 10)
    results_summary = "\n\n".join(
        f"Task {h.key} — {h.value['description']}:\n{_truncate_for_validation(h.value['result'])}"
        for h in hits
    ) or "No tasks completed."

    failure_summary = "\n".join(
        f"Task {t['task_id']} — {t['description']}: failed after 3 attempts" for t in failed
    )

    guidance_prompt = SystemMessage(content="""You are a summarization agent. Given the results of completed tasks, produce a clear and concise final response to the original request.
If some tasks failed, acknowledge what could not be completed without dwelling on it.""")

    action_prompt = HumanMessage(content=f"""Original request: {state["prompt"]}

Completed task results:
{results_summary}

{"Failed tasks:" + chr(10) + failure_summary if failed else ""}""")

    global INPUT_TOKENS, OUTPUT_TOKENS
    _stop_agent_running_ticker()
    status.update(status="Thinking...", spinner="aesthetic")
    with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
        result = llm.invoke([guidance_prompt, action_prompt])
    status.update(status="Complete", spinner="bouncingBall")
    status.stop()
    INPUT_TOKENS += result.usage_metadata["input_tokens"]
    OUTPUT_TOKENS += result.usage_metadata["output_tokens"]

    console.print(Markdown(result.content))

    execution_time = time() - START_TIME
    total_tokens = INPUT_TOKENS + OUTPUT_TOKENS

    log("Done.")
    log(f"Total tokens — input: {INPUT_TOKENS:,}  output: {OUTPUT_TOKENS:,}")

    return {
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "total_tokens": total_tokens,
        "tokens_sec": 0,
        "execution_time": int(execution_time)
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_task_manager(state: AgentState) -> str:
    return "agent_terminate" if state["termination_reason"] else "agent_execute"


def route_validate(state: AgentState) -> str:
    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])
    return "agent_execute" if current["status"] == "in_progress" else "task_manager"


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(AgentState, input_schema=StartState, output_schema=EndState)
builder.add_node("agent_bootstrap", agent_bootstrap)
builder.add_node("agent_plan", planning_agent)
builder.add_node("task_manager", task_manager)
builder.add_node("agent_execute", agent_execute)
builder.add_node("agent_validate", agent_validate)
builder.add_node("agent_terminate", agent_terminate)

builder.add_edge(START, "agent_bootstrap")
builder.add_edge("agent_bootstrap", "agent_plan")
builder.add_edge("agent_plan", "task_manager")
builder.add_conditional_edges("task_manager", route_task_manager)
builder.add_edge("agent_execute", "agent_validate")
builder.add_conditional_edges("agent_validate", route_validate)
builder.add_edge("agent_terminate", END)

agent = builder.compile(store=memory_store)


if __name__ == "__main__":
    import argparse
    from rich.markdown import Markdown

    parser = argparse.ArgumentParser(description="Run the agent with an optional prompt.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Task prompt for the agent. If omitted, a default prompt is used.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM cache for this run (fresh API calls).",
    )

    args = parser.parse_args()

    if args.prompt:
        start_prompt = args.prompt

    else:
        start_prompt = """
    list files in this folder
    read each python file found in the list
    summarize each python app
    """

    console.print(agent.invoke({"start_prompt": start_prompt}))
