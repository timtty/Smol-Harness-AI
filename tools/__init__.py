"""
Tools package: web, bash, file_ops, and agent task tools.
Import all_tools, file_ops_tooling, website_ops_tooling, add_task from here or from submodules.
"""

from tools.web import website_ops_tooling
from tools.file_ops import file_ops_tooling, find_lines
from tools.bash import bash_execute
from tools.agent import add_task

file_ops_tooling = list(file_ops_tooling)  # copy so agents can extend
# file_ops includes read_file, write_file; bash is separate
all_file_and_bash_tools = [*file_ops_tooling, bash_execute]
website_ops_tooling = list(website_ops_tooling)

all_tools = all_file_and_bash_tools + website_ops_tooling + [add_task]
tool_map = {t.name: t for t in all_tools}

__all__ = [
    "website_ops_tooling",
    "file_ops_tooling",
    "all_file_and_bash_tools",
    "bash_execute",
    "find_lines",
    "add_task",
    "all_tools",
    "tool_map",
]
