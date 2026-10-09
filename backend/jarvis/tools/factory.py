"""Build the tool kit from settings."""

from __future__ import annotations

from jarvis.config import Settings
from jarvis.memory.store import MemoryStore
from jarvis.tools.audit import AuditLog
from jarvis.tools.base import Policy, Tool
from jarvis.tools.builtin.basic import CalculatorTool, ClockTool
from jarvis.tools.builtin.files import FileAccess, ListDirTool, ReadFileTool
from jarvis.tools.builtin.python_sandbox import PythonSandboxTool
from jarvis.tools.builtin.tasks import AddTaskTool, CompleteTaskTool, DeleteTaskTool, ListTasksTool
from jarvis.tools.builtin.web import FetchUrlTool, WikipediaTool
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.toolkit import ToolKit


def all_tools(settings: Settings, memory: MemoryStore | None) -> list[Tool]:
    access = FileAccess(settings.tools.allowed_dirs)
    return [
        CalculatorTool(),
        ClockTool(),
        WikipediaTool(),
        FetchUrlTool(),
        AddTaskTool(memory),
        ListTasksTool(memory),
        CompleteTaskTool(memory),
        DeleteTaskTool(memory),
        ListDirTool(access),
        ReadFileTool(access),
        PythonSandboxTool(),
    ]


def build_toolkit(settings: Settings, memory: MemoryStore | None) -> ToolKit | None:
    cfg = settings.tools
    if not cfg.enabled:
        return None
    policies = {name: Policy(value) for name, value in cfg.policies.items()}
    executor = ToolExecutor(
        all_tools(settings, memory),
        policies=policies,
        audit=AuditLog(settings.data_dir / "audit.db"),
        max_calls_per_minute=cfg.max_calls_per_minute,
    )
    return ToolKit(executor, max_steps=cfg.max_steps)
