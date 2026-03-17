"""Entry point for the `smolee` CLI command."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Run the smol-harness agent.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Task prompt for the agent. If omitted, a default prompt is used.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable search cache for this run (fresh API calls).",
    )
    args = parser.parse_args()

    # Ensure the project root is on sys.path so basic_agent is importable
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from basic_agent import agent, console

    start_prompt = args.prompt or """
    list files in this folder
    read each python file found in the list
    summarize each python app
    """

    agent.invoke({"start_prompt": start_prompt})
