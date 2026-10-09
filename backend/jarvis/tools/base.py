"""Tool contract: what a tool declares about itself and what it returns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class Risk(IntEnum):
    SAFE = 0  # pure computation, no data leaves or changes (calculator, clock)
    SENSITIVE = 1  # reads your data or the internet, or changes JARVIS's own data
    DANGEROUS = 2  # runs code or could change things outside JARVIS


class Policy(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class ToolError(Exception):
    """Expected failure (bad argument, path not allowed…): the agent sees the message."""


@dataclass(frozen=True)
class Param:
    name: str
    description: str
    required: bool = True


@dataclass
class ToolResult:
    text: str
    ok: bool = True


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


ConfirmFn = Callable[["ToolCall", "Tool", str], Awaitable[bool]]
EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class ToolContext:
    """Per-request context: how to ask the user, and what happened so far."""

    confirm: ConfirmFn | None = None  # None: tools needing confirmation are denied
    on_event: EventFn | None = None
    conversation_id: str | None = None
    agent_id: str | None = None
    tainted: bool = False  # untrusted content (web page, file) entered this request
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)


class Tool(ABC):
    name: str  # identifier the agent uses
    title: str  # shown to the user (Portuguese)
    description: str  # shown to the agent
    params: tuple[Param, ...] = ()
    risk: Risk = Risk.SAFE
    default_policy: Policy = Policy.ALLOW
    untrusted_output: bool = False  # the result may contain text written by third parties
    timeout: float = 15.0

    def available(self) -> str | None:
        """None when the tool can run; otherwise the reason it cannot."""
        return None

    def describe_call(self, args: dict[str, Any]) -> str:
        """One line telling the user what this call will do (for confirmations and logs)."""
        shown = ", ".join(f"{k}={str(v)[:120]!r}" for k, v in args.items())
        return f"{self.title}({shown})"

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    @staticmethod
    def arg(args: dict[str, Any], name: str, *, max_len: int = 2000) -> str:
        value = args.get(name)
        if not isinstance(value, (str, int, float)) or str(value).strip() == "":
            raise ToolError(f"argumento obrigatório ausente: {name}")
        text = str(value).strip()
        if len(text) > max_len:
            raise ToolError(f"argumento {name} grande demais (máximo {max_len} caracteres)")
        return text
