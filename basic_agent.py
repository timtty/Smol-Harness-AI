from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

from langchain.messages import SystemMessage
from langchain.messages import HumanMessage
from langchain.tools import tool

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict
from typing_extensions import List
from typing_extensions import Annotated
from shortuuid import uuid as suid

from json import dumps as json_to_string
from json import loads as json_parse

from subprocess import run as subprocess_run
from subprocess import PIPE
from subprocess import TimeoutExpired

from requests import get as http_get
from html2text import HTML2Text
from xml.etree.ElementTree import fromstring as xml_parse
from bs4 import BeautifulSoup
import re

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

# Max chars of task result to send to validator (avoids context overflow from long webpages/files)
VALIDATE_RESULT_MAX_CHARS = 24_000
# Max chars per result when building context for summarize step (5 results × this = total cap)
SUMMARIZE_RESULT_MAX_CHARS = 20_000
# Max chars returned from read_webpage to avoid huge state and store
READ_WEBPAGE_MAX_CHARS = 100_000
# When cleaned text exceeds this, use sumy (TextRank) extractive summary instead of raw truncation
SUMMY_SUMMARIZE_WHEN_OVER_CHARS = 25_000
# Max sentences to keep when sumy summarization runs
SUMMY_MAX_SENTENCES = 80

# Elements to strip from HTML before conversion (noise, non-content)
_STRIP_HTML_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "object", "embed",
    "nav", "header", "footer", "aside", "form", "button", "input", "select", "textarea",
)
# Roles that usually wrap chrome, not main content
_STRIP_HTML_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search")


def _strip_html_noise(html: str) -> str:
    """Remove script, style, nav, and other typical noise before converting to text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_STRIP_HTML_TAGS):
        tag.decompose()
    for el in soup.find_all(attrs={"role": lambda v: v and v in _STRIP_HTML_ROLES}):
        el.decompose()
    # Prefer main content if present and substantial
    main = soup.find("main") or soup.find("article")
    if main and len(main.get_text(strip=True)) > 200:
        body = main
    else:
        body = soup.find("body") or soup
    return str(body)


def _clean_extracted_text(text: str) -> str:
    """Collapse excessive whitespace and drop common UI-only lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    skip_phrases = re.compile(
        r"^(share|tweet|subscribe|newsletter|cookie|accept all|manage preferences|"
        r"follow us|related articles|read more|advertisement|sponsored|\\|/|·)$",
        re.I,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or skip_phrases.match(line) or len(line) <= 2:
            continue
        lines.append(line)
    return "\n\n".join(lines)


def _extractive_summary(text: str, max_sentences: int = SUMMY_MAX_SENTENCES) -> str:
    """Use sumy TextRank to reduce long text to top-ranked sentences (no LLM)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        n = min(max_sentences, len(parser.document.sentences))
        if n <= 0:
            return text
        summary_sentences = summarizer(parser.document, n)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        return text


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


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(description="Search the web for information")
def website_search(search_term: str) -> str:
    response = http_get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": env["BRAVE_SEARCH_API_KEY"]},
        params={"q": search_term},
    )

    if response.status_code == 200:
        results = response.json().get("web", {}).get("results", [])
        return json_to_string([{"title": r["title"], "url": r["url"], "description": r.get("description", "")} for r in results])

    else:
        return f"Search failed: {response.status_code}"


@tool(description="Read and retrieve the content of a webpage by URL")
def read_webpage(url: str) -> str:
    try:
        response = http_get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html_clean = _strip_html_noise(response.text)
        h = HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        text = h.handle(html_clean)
        text = _clean_extracted_text(text)
        if len(text) > SUMMY_SUMMARIZE_WHEN_OVER_CHARS:
            text = _extractive_summary(text, SUMMY_MAX_SENTENCES)
            if len(text) > READ_WEBPAGE_MAX_CHARS:
                text = text[:READ_WEBPAGE_MAX_CHARS] + "\n\n[... content truncated to fit context ...]"
        elif len(text) > READ_WEBPAGE_MAX_CHARS:
            text = text[:READ_WEBPAGE_MAX_CHARS] + "\n\n[... content truncated to fit context ...]"
        return text
    except Exception as e:
        return f"Failed to fetch page: {e}"


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


@tool(description="Execute a bash command and return its stdout, stderr, and exit code. Use for local system operations, running scripts, listing files, etc. Avoid long-running or interactive commands.")
def bash_execute(command: str, timeout_seconds: int = 30) -> str:
    try:
        proc = subprocess_run(
            command,
            shell=True,
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
        return "\n".join(parts)
    except TimeoutExpired:
        return f"Command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"Error executing command: {e}"


file_ops_tooling = [
    read_file,
    write_file,
    bash_execute,
]

@tool(description="Fetch an RSS feed and return article titles, descriptions, and URLs. Prefer this over read_webpage when the URL ends in .xml or is a known RSS/Atom feed.")
def fetch_rss_articles(feed_url: str, max_article_count: int = 5) -> str:
    try:
        response = http_get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = xml_parse(response.content)
        items = root.findall(".//item")
        articles = []
        for item in items[:max_article_count]:
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            link  = (item.findtext("link") or "").strip()
            articles.append({"title": title, "description": desc[:300], "url": link})
        return json_to_string(articles)
    except Exception as e:
        return f"Failed to fetch RSS feed: {e}"


website_ops_tooling = [
    website_search,
    read_webpage,
    fetch_rss_articles,
]

_pending_tasks: List[dict] = []


@tool(description="Queue a new task to be executed. Use this when you discover new work during execution, such as finding URLs that each need to be read separately.")
def add_task(description: str, action: str, input_hint: str) -> str:
    _pending_tasks.append({
        "description": description,
        "action": action,
        "input_hint": input_hint,
    })
    return f"Task queued: {description}"


all_tools = file_ops_tooling + website_ops_tooling + [add_task]
tool_map = {t.name: t for t in all_tools}


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
- bash_execute      — run a bash command and capture stdout/stderr/exit code; use for local system tasks, file listing, running scripts, etc.
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


def agent_execute(state: AgentState, *, store: BaseStore) -> AgentState:
    global INPUT_TOKENS, OUTPUT_TOKENS
    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])

    retry_suffix = f" (retry {current['retry_count']}/3)" if current["retry_count"] > 0 else ""
    log(f"Executing task {current['task_id']}/{len(state['tasks'])}: {current['description']}{retry_suffix}")

    retry_context = ""
    if current["error_context"]:
        retry_context = "\n\nPrevious attempt errors:\n" + "\n".join(current["error_context"])

    if current["action"] == "summarize":
        llm = _llm()
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
If prior task results are provided, extract URLs or data from them rather than performing a new search.
If during execution you discover new work items (e.g. multiple URLs that each need to be read), use add_task to queue them rather than trying to handle everything yourself.""")
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

    if _pending_tasks:
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
            for i, t in enumerate(_pending_tasks)
        ]
        _pending_tasks.clear()
        updated_tasks = updated_tasks + new_tasks
        log(f"Added {len(new_tasks)} new task(s) to the plan")
        print_task_list(updated_tasks)

    return {"tasks": updated_tasks}


def agent_validate(state: AgentState, *, store: BaseStore) -> AgentState:
    global INPUT_TOKENS, OUTPUT_TOKENS
    llm = _llm()
    validator = llm.with_structured_output(ValidationResult, include_raw=True)

    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])

    log(f"Validating task {current['task_id']}...")

    guidance_prompt = SystemMessage(content="""You are a validation agent. Assess whether a task was completed successfully.
Return valid=true if the result meaningfully addresses the task description.
Return valid=false with a reason if the result is empty, an error, irrelevant, or clearly incomplete.""")

    result_for_prompt = _truncate_for_validation(current["result"])
    action_prompt = HumanMessage(content=f"Task: {current['description']}\nResult: {result_for_prompt}")

    _stop_agent_running_ticker()
    status.update(status="Thinking...", spinner="aesthetic")
    with _thinking_with_elapsed_ticker(status, min_seconds_before_ticker=3):
        verdict = validator.invoke([guidance_prompt, action_prompt])
    status.update(status="Agent is running...", spinner="bouncingBall")
    _start_agent_running_ticker(status)
    INPUT_TOKENS += verdict["raw"].usage_metadata["input_tokens"]
    OUTPUT_TOKENS += verdict["raw"].usage_metadata["output_tokens"]

    if verdict["parsed"]["valid"]:
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
        log(f"Task {current['task_id']} failed validation ({new_retry_count}/3): {verdict['parsed']['reason']}")
        updated_tasks = [
            {**t, "status": "in_progress", "retry_count": new_retry_count, "error_context": current["error_context"] + [verdict["parsed"]["reason"]]}
            if t["task_id"] == state["current_task_id"] else t
            for t in state["tasks"]
        ]

    print_task_list(updated_tasks)
    return {"tasks": updated_tasks}


def agent_terminate(state: AgentState, *, store: BaseStore) -> EndState:
    llm = _llm()

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
