from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

from langchain.messages import SystemMessage
from langchain.messages import HumanMessage
from langchain.tools import tool

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing_extensions import List
from typing_extensions import Annotated
from shortuuid import uuid as suid

from json import dumps as json_to_string
from json import loads as json_parse

from requests import get as http_get
from html2text import HTML2Text
from xml.etree.ElementTree import fromstring as xml_parse

from dotenv import load_dotenv
from os import environ as env
from random import choice as random_choice
from time import time
from time import sleep

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.tree import Tree
from rich import box



console = Console()
console.clear()
status = console.status("Waking up...")

load_dotenv()

PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#E8BAFF", "#FFB3E6", "#B3F0FF", "#C9FFE5", "#FFD9B3",
]

def log(message: str) -> None:
    color = random_choice(PASTEL_COLORS)
    console.print(f"[{color}]●[/] {message}\n")

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        base_url=env.get("OPENAI_BASE_URL"),
        api_key=env.get("OPENAI_API_KEY"),
    )


MAX_ITERATIONS = 20
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0
START_TIME = time()
MODEL = "qwen/qwen3-4b-2507"
CONTEXT_WINDOW_MAX = 100_000


def _ctx_status() -> str:
    used = INPUT_TOKENS + OUTPUT_TOKENS
    pct = int(used / CONTEXT_WINDOW_MAX * 100)
    return f"Agent is running... (context window: {used // 1000}k/{CONTEXT_WINDOW_MAX // 1000}k - {pct}%)"


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
        h = HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        return h.handle(response.text)
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


file_ops_tooling = [
    read_file,
    write_file,
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
    agent_response: str


class Task(TypedDict):
    task_id: int
    description: str
    action: str            # website_search | read_webpage | fetch_rss_articles | read_file | write_file | summarize
    input_hint: str
    status: str            # pending | in_progress | complete | failed
    result: str
    retry_count: int
    error_context: List[str]


class AgentState(TypedDict):
    prompt: str
    tasks: List[Task]
    current_task_id: int
    iterations: int
    termination_reason: str


# ── Planner structured output schemas ─────────────────────────────────────────

class PlanStep(TypedDict):
    step_id: int
    description: str
    action: str      # website_search | read_webpage | fetch_rss_articles | read_file | write_file | summarize
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
    status.update(status="Thinking...")
    result = planner.invoke([guidance_prompt, action_prompt])
    status.update(status=_ctx_status())

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
        completed_results = "\n\n".join(
            f"Task {t['task_id']} — {t['description']}:\n{t['result']}"
            for t in state["tasks"] if t["status"] == "complete"
        )
        guidance_prompt = SystemMessage(content="You are a summarization agent. Synthesize the provided information into a clear, concise summary.")
        action_prompt = HumanMessage(content=f"Task: {current['description']}\n\nContext:\n{completed_results}{retry_context}")
        llm_response = llm.invoke([guidance_prompt, action_prompt])
        INPUT_TOKENS += llm_response.usage_metadata["input_tokens"]
        OUTPUT_TOKENS += llm_response.usage_metadata["output_tokens"]
        result = llm_response.content

    else:
        llm = _llm()
        executor = llm.bind_tools(all_tools)
        guidance_prompt = SystemMessage(content="""You are a task execution agent. Use the available tools to complete the task.
If prior task results are provided, extract URLs or data from them rather than performing a new search.
If during execution you discover new work items (e.g. multiple URLs that each need to be read), use add_task to queue them rather than trying to handle everything yourself.""")
        completed = [t for t in state["tasks"] if t["status"] == "complete"]
        prior_context = ""
        if completed:
            prior_context = "\n\nCompleted task results:\n" + "\n\n".join(
                f"Task {t['task_id']} — {t['description']}:\n{t['result']}"
                for t in completed
            )
        action_prompt = HumanMessage(content=f"Task: {current['description']}\nSuggested input: {current['input_hint']}{prior_context}{retry_context}")
        status.update(status="Thinking...")
        response = executor.invoke([guidance_prompt, action_prompt])
        status.update(status=_ctx_status())
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


def agent_validate(state: AgentState) -> AgentState:
    global INPUT_TOKENS, OUTPUT_TOKENS
    llm = _llm()
    validator = llm.with_structured_output(ValidationResult, include_raw=True)

    current = next(t for t in state["tasks"] if t["task_id"] == state["current_task_id"])

    log(f"Validating task {current['task_id']}...")

    guidance_prompt = SystemMessage(content="""You are a validation agent. Assess whether a task was completed successfully.
Return valid=true if the result meaningfully addresses the task description.
Return valid=false with a reason if the result is empty, an error, irrelevant, or clearly incomplete.""")

    action_prompt = HumanMessage(content=f"Task: {current['description']}\nResult: {current['result']}")

    status.update(status="Thinking...")
    verdict = validator.invoke([guidance_prompt, action_prompt])
    status.update(status=_ctx_status())
    INPUT_TOKENS += verdict["raw"].usage_metadata["input_tokens"]
    OUTPUT_TOKENS += verdict["raw"].usage_metadata["output_tokens"]

    if verdict["parsed"]["valid"]:
        log(f"Task {current['task_id']} passed validation")
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


def agent_terminate(state: AgentState) -> EndState:
    llm = _llm()

    completed = [t for t in state["tasks"] if t["status"] == "complete"]
    failed = [t for t in state["tasks"] if t["status"] == "failed"]

    log(f"Terminating ({state['termination_reason']}): {len(completed)} complete, {len(failed)} failed. Generating final response...")

    results_summary = "\n\n".join(
        f"Task {t['task_id']} — {t['description']}:\n{t['result']}" for t in completed
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
    status.update(status="Thinking...")
    result = llm.invoke([guidance_prompt, action_prompt])
    status.update(status="Complete")
    status.stop()
    INPUT_TOKENS += result.usage_metadata["input_tokens"]
    OUTPUT_TOKENS += result.usage_metadata["output_tokens"]

    log("Done.")
    log(f"Total tokens — input: {INPUT_TOKENS:,}  output: {OUTPUT_TOKENS:,}")
    return {"agent_response": result.content}


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

agent = builder.compile()


if __name__ == "__main__":
    start_prompt = """
    find the official NYT RSS feed *list* via search.
    read only the NYT RSS list then choose 3 categories and their URLs from this list not your memory.
    read an article from each category and return a brief summary of each article.
    finally provide a final summary of all 3 summaries
    """
    console.print(agent.invoke({"start_prompt": start_prompt}))
