"""Shared data types for agents, requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Capability(StrEnum):
    CHAT = "chat"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    RESEARCH = "research"
    VISION = "vision"


class TaskType(StrEnum):
    CHAT = "chat"
    CODE = "code"
    MATH = "math"
    RESEARCH = "research"


# The capability that best serves each task type.
TASK_CAPABILITY: dict[TaskType, Capability] = {
    TaskType.CHAT: Capability.CHAT,
    TaskType.CODE: Capability.CODE,
    TaskType.MATH: Capability.MATH,
    TaskType.RESEARCH: Capability.RESEARCH,
}


class CostClass(StrEnum):
    """How an agent is paid for. Only PAID can ever produce a charge."""

    LOCAL = "local"  # runs on this machine
    FREE = "free"  # free model, no billing account attached
    FREE_WITH_LIMITS = "free_with_limits"  # free quota that resets, no card on file
    PAID = "paid"  # may charge money: blocked unless explicitly allowed


class CostMode(StrEnum):
    FREE_ONLY = "FREE_ONLY"
    ALLOW_PAID = "ALLOW_PAID"


class Privacy(StrEnum):
    LOCAL = "local"
    ZERO_RETENTION = "zero_retention"
    MAY_TRAIN = "may_train"
    UNKNOWN = "unknown"


class Health(StrEnum):
    AVAILABLE = "available"
    COOLDOWN = "cooldown"  # temporarily limited (rate limit or repeated errors)
    OFFLINE = "offline"  # health check failed
    UNCONFIGURED = "unconfigured"  # missing key or disabled in config
    BLOCKED = "blocked"  # cost guard refused it; never used again this run
    UNKNOWN = "unknown"  # not checked yet


class AgentInfo(BaseModel):
    """Static description of an agent, declared by its adapter."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    provider: str
    model: str
    cost_class: CostClass
    is_local: bool = False
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    priority: int = 50  # 0..100, higher is preferred
    privacy: Privacy = Privacy.UNKNOWN
    daily_limit: int | None = None  # known requests/day quota, if any
    allow_paid: bool = False  # explicit per-agent opt-in, only honoured in ALLOW_PAID mode


class AgentStatus(BaseModel):
    """Runtime view of an agent, as exposed to the UI and the CLI."""

    id: str
    name: str
    provider: str
    model: str
    health: Health
    available: bool
    remaining_usage: int | None = None
    rate_limit: int | None = None
    cooldown_until: datetime | None = None
    last_error: str | None = None
    capabilities: list[Capability]
    priority: int
    is_local: bool
    requires_payment: bool
    privacy: Privacy
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float | None = None
    requests_today: int = 0


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AIRequest(BaseModel):
    messages: list[Message]
    task: TaskType = TaskType.CHAT
    required: frozenset[Capability] = frozenset()  # hard requirements (e.g. vision)
    max_output_tokens: int | None = None


class AIResponse(BaseModel):
    text: str
    agent_id: str
    model: str
    cost: float | None = None  # USD as reported by the provider, when it reports one
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0


class Attempt(BaseModel):
    agent_id: str
    ok: bool
    error: str | None = None
    latency_ms: float = 0.0


class OrchestratorResult(BaseModel):
    response: AIResponse
    task: TaskType
    attempts: list[Attempt] = Field(default_factory=list)
