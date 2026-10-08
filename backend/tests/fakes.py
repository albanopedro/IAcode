"""Test doubles: agents and a clock that never touch the network."""

from __future__ import annotations

from jarvis.core.provider import AIProvider, HealthReport
from jarvis.core.types import (
    AgentInfo,
    AIRequest,
    AIResponse,
    Capability,
    CostClass,
    Health,
    Privacy,
)


class FakeClock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAgent(AIProvider):
    """Answers from a script: each item is a reply text or an exception to raise."""

    def __init__(
        self,
        agent_id: str,
        script: list | None = None,
        *,
        priority: int = 50,
        capabilities: set[Capability] | None = None,
        cost_class: CostClass = CostClass.FREE,
        privacy: Privacy = Privacy.UNKNOWN,
        daily_limit: int | None = None,
        health: Health = Health.AVAILABLE,
        cost: float | None = 0.0,
        latency_ms: float = 100.0,
        allow_paid: bool = False,
        rpm_limit: int | None = None,
        quota_group: str | None = None,
        quality: dict | None = None,
        context_window: int | None = None,
        rate_limit=None,
    ) -> None:
        self.info = AgentInfo(
            id=agent_id,
            name=agent_id,
            provider="fake",
            model=f"{agent_id}-model",
            cost_class=cost_class,
            capabilities=frozenset(capabilities or {Capability.CHAT}),
            priority=priority,
            privacy=privacy,
            daily_limit=daily_limit,
            allow_paid=allow_paid,
            rpm_limit=rpm_limit,
            quota_group=quota_group,
            quality=quality or {},
            context_window=context_window,
        )
        self.rate_limit = rate_limit
        self.script = list(script or [f"resposta de {agent_id}"])
        self.health = health
        self.cost = cost
        self.latency_ms = latency_ms
        self.calls: list[AIRequest] = []
        self.health_checks = 0
        self.closed = False

    async def check_health(self) -> HealthReport:
        self.health_checks += 1
        return HealthReport(self.health, "fake")

    async def generate(self, request: AIRequest) -> AIResponse:
        self.calls.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return AIResponse(
            text=item,
            agent_id=self.id,
            model=self.info.model,
            cost=self.cost,
            latency_ms=self.latency_ms,
            rate_limit=self.rate_limit,
        )

    async def close(self) -> None:
        self.closed = True
