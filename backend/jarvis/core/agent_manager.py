"""Agent manager: registry of agents plus their runtime state.

It answers "who can take a request right now?" and learns from every call:
rate limits become cooldowns, repeated failures back off exponentially, and
billing problems block an agent for the rest of the run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import (
    CostViolationError,
    NotConfiguredError,
    PaymentRequiredError,
    ProviderError,
    RateLimitError,
)
from jarvis.core.provider import AIProvider, HealthReport
from jarvis.core.types import AgentStatus, AIResponse, Capability, CostClass, Health

DEFAULT_RATE_LIMIT_COOLDOWN = 60.0
BACKOFF_BASE = 15.0
BACKOFF_MAX = 600.0
HEALTH_TTL = 60.0
LATENCY_ALPHA = 0.3  # weight of the newest sample in the moving average


@dataclass
class RuntimeState:
    health: Health = Health.UNKNOWN
    cooldown_until: float | None = None
    last_error: str | None = None
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float | None = None
    requests_today: int = 0
    day: str = ""
    checked_at: float | None = None


class AgentManager:
    def __init__(
        self,
        providers: Iterable[AIProvider],
        cost_guard: CostGuard,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cost_guard = cost_guard
        self._clock = clock
        self._providers: dict[str, AIProvider] = {}
        self._state: dict[str, RuntimeState] = {}
        for provider in providers:
            self.register(provider)

    # -- registry -----------------------------------------------------------

    def register(self, provider: AIProvider) -> None:
        if provider.id in self._providers:
            raise ValueError(f"duplicate agent id: {provider.id}")
        self._providers[provider.id] = provider
        state = RuntimeState()
        ok, reason = self.cost_guard.allowed(provider.info)
        if not ok:
            state.health = Health.BLOCKED
            state.last_error = reason
        self._state[provider.id] = state

    def get(self, agent_id: str) -> AIProvider:
        return self._providers[agent_id]

    @property
    def providers(self) -> list[AIProvider]:
        return list(self._providers.values())

    def state(self, agent_id: str) -> RuntimeState:
        return self._state[agent_id]

    # -- health ---------------------------------------------------------------

    async def refresh_health(self, *, force: bool = False) -> None:
        now = self._clock()
        due = [
            p
            for p in self._providers.values()
            if self._state[p.id].health is not Health.BLOCKED
            and (force or self._is_stale(p.id, now))
        ]
        reports = await asyncio.gather(*(self._safe_check(p) for p in due))
        for provider, report in zip(due, reports, strict=True):
            state = self._state[provider.id]
            state.checked_at = now
            if state.health is Health.BLOCKED:
                continue
            if report.health is Health.AVAILABLE:
                # A healthy check does not end an active cooldown.
                state.health = (
                    Health.COOLDOWN if self._in_cooldown(state, now) else Health.AVAILABLE
                )
            else:
                state.health = report.health
                state.last_error = report.detail or state.last_error

    def _is_stale(self, agent_id: str, now: float) -> bool:
        checked = self._state[agent_id].checked_at
        return checked is None or now - checked > HEALTH_TTL

    @staticmethod
    async def _safe_check(provider: AIProvider) -> HealthReport:
        try:
            return await provider.check_health()
        except Exception as exc:  # a broken health check must not break JARVIS
            return HealthReport(Health.OFFLINE, f"health check crashed: {exc}")

    # -- selection ------------------------------------------------------------

    def candidates(self, required: frozenset[Capability] = frozenset()) -> list[AIProvider]:
        now = self._clock()
        result = []
        for provider in self._providers.values():
            state = self._state[provider.id]
            self._roll_day(state, now)
            if state.health is Health.COOLDOWN and not self._in_cooldown(state, now):
                state.health = Health.AVAILABLE
            if state.health is not Health.AVAILABLE:
                continue
            if not all(provider.supports(cap) for cap in required):
                continue
            result.append(provider)
        return result

    # -- learning from calls -------------------------------------------------

    def record_success(self, agent_id: str, response: AIResponse) -> None:
        state = self._state[agent_id]
        now = self._clock()
        self._roll_day(state, now)
        state.successes += 1
        state.consecutive_failures = 0
        state.requests_today += 1
        state.last_error = None
        if state.avg_latency_ms is None:
            state.avg_latency_ms = response.latency_ms
        else:
            state.avg_latency_ms = (
                LATENCY_ALPHA * response.latency_ms + (1 - LATENCY_ALPHA) * state.avg_latency_ms
            )
        limit = self._providers[agent_id].info.daily_limit
        if limit is not None and state.requests_today >= limit:
            self._cooldown(state, self._seconds_until_utc_midnight(now), "daily quota used up")

    def record_failure(self, agent_id: str, error: ProviderError) -> None:
        state = self._state[agent_id]
        now = self._clock()
        self._roll_day(state, now)
        state.failures += 1
        state.consecutive_failures += 1
        state.last_error = str(error)
        state.requests_today += 1

        if isinstance(error, (PaymentRequiredError, CostViolationError)):
            state.health = Health.BLOCKED
            state.cooldown_until = None
        elif isinstance(error, NotConfiguredError):
            state.health = Health.UNCONFIGURED
        elif isinstance(error, RateLimitError):
            self._cooldown(state, error.retry_after or DEFAULT_RATE_LIMIT_COOLDOWN, str(error))
        else:
            delay = min(BACKOFF_BASE * 2 ** (state.consecutive_failures - 1), BACKOFF_MAX)
            if error.retry_after:
                delay = max(delay, error.retry_after)
            self._cooldown(state, delay, str(error))

    def _cooldown(self, state: RuntimeState, seconds: float, reason: str) -> None:
        state.health = Health.COOLDOWN
        state.cooldown_until = self._clock() + seconds
        state.last_error = reason

    def _in_cooldown(self, state: RuntimeState, now: float) -> bool:
        return state.cooldown_until is not None and now < state.cooldown_until

    def _roll_day(self, state: RuntimeState, now: float) -> None:
        day = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        if state.day != day:
            state.day = day
            state.requests_today = 0

    @staticmethod
    def _seconds_until_utc_midnight(now: float) -> float:
        current = datetime.fromtimestamp(now, UTC)
        midnight = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (midnight - current).total_seconds()

    # -- reporting ------------------------------------------------------------

    def status(self, agent_id: str) -> AgentStatus:
        provider = self._providers[agent_id]
        info = provider.info
        state = self._state[agent_id]
        now = self._clock()
        if state.health is Health.COOLDOWN and not self._in_cooldown(state, now):
            state.health = Health.AVAILABLE
        remaining = None
        if info.daily_limit is not None:
            remaining = max(info.daily_limit - state.requests_today, 0)
        cooldown = None
        if self._in_cooldown(state, now):
            cooldown = datetime.fromtimestamp(state.cooldown_until, UTC)
        return AgentStatus(
            id=info.id,
            name=info.name,
            provider=info.provider,
            model=info.model,
            health=state.health,
            available=state.health is Health.AVAILABLE,
            remaining_usage=remaining,
            rate_limit=info.daily_limit,
            cooldown_until=cooldown,
            last_error=state.last_error,
            capabilities=sorted(info.capabilities),
            priority=info.priority,
            is_local=info.is_local,
            requires_payment=info.cost_class is CostClass.PAID,
            privacy=info.privacy,
            successes=state.successes,
            failures=state.failures,
            consecutive_failures=state.consecutive_failures,
            avg_latency_ms=state.avg_latency_ms,
            requests_today=state.requests_today,
        )

    def statuses(self) -> list[AgentStatus]:
        return [self.status(agent_id) for agent_id in self._providers]

    async def close(self) -> None:
        await asyncio.gather(*(p.close() for p in self._providers.values()), return_exceptions=True)
