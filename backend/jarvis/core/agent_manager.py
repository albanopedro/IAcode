"""Agent manager: registry of agents plus their runtime state.

It answers "who can take a request right now?" and learns from every call:
- rate limits become cooldowns (using the provider's own Retry-After / quota headers);
- a known requests-per-minute quota is respected locally, before the provider says 429;
- daily quotas are counted and persisted, so a restart does not forget them;
- repeated failures back off exponentially (15 s → 10 min), like a circuit breaker:
  when the cooldown ends the agent gets one new chance, and fails slower next time;
- billing problems block an agent for good — persisted until ``jarvis unblock``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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
from jarvis.core.types import (
    AgentStatus,
    AIResponse,
    Capability,
    CostClass,
    Health,
    RateLimitInfo,
)
from jarvis.core.usage_store import UsageStore

DEFAULT_RATE_LIMIT_COOLDOWN = 60.0
BACKOFF_BASE = 15.0
BACKOFF_MAX = 600.0
HEALTH_TTL = 60.0
LATENCY_ALPHA = 0.3  # weight of the newest sample in the moving average
OUTCOME_WINDOW = 20  # calls used for the success rate
MINUTE = 60.0


@dataclass
class RuntimeState:
    health: Health = Health.UNKNOWN
    cooldown_until: float | None = None
    blocked_reason: str | None = None  # runtime block (cost/billing), persisted
    last_error: str | None = None
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float | None = None
    requests_today: int = 0
    day: str = ""
    checked_at: float | None = None
    minute_window: deque[float] = field(default_factory=deque)
    outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=OUTCOME_WINDOW))
    provider_quota: RateLimitInfo | None = None

    @property
    def success_rate(self) -> float | None:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else None


def utc_day(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")


class AgentManager:
    def __init__(
        self,
        providers: Iterable[AIProvider],
        cost_guard: CostGuard,
        clock: Callable[[], float] = time.time,
        store: UsageStore | None = None,
    ) -> None:
        self.cost_guard = cost_guard
        self.store = store
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
        now = self._clock()
        state.day = utc_day(now)
        self._state[provider.id] = state

        ok, reason = self.cost_guard.allowed(provider.info)
        if not ok:
            state.health = Health.BLOCKED
            state.last_error = reason
            return
        if self.store is None:
            return
        stored = self.store.load(provider.id, state.day)
        state.requests_today = stored.requests_today
        state.last_error = stored.last_error
        if stored.blocked_reason:
            state.health = Health.BLOCKED
            state.blocked_reason = stored.blocked_reason
            state.last_error = (
                f"bloqueado numa execução anterior: {stored.blocked_reason} "
                f"(para liberar: jarvis unblock {provider.id})"
            )
        elif stored.cooldown_until and stored.cooldown_until > now:
            state.health = Health.COOLDOWN
            state.cooldown_until = stored.cooldown_until

    def _group_key(self, agent_id: str) -> str:
        return self._providers[agent_id].info.quota_group or agent_id

    def _group(self, agent_id: str) -> list[str]:
        key = self._group_key(agent_id)
        return [a for a in self._providers if self._group_key(a) == key]

    def _group_requests_today(self, agent_id: str) -> int:
        return sum(self._state[a].requests_today for a in self._group(agent_id))

    def _group_window(self, agent_id: str, now: float) -> list[float]:
        times = sorted(t for a in self._group(agent_id) for t in self._state[a].minute_window)
        return [t for t in times if now - t < MINUTE]

    def get(self, agent_id: str) -> AIProvider:
        return self._providers[agent_id]

    @property
    def providers(self) -> list[AIProvider]:
        return list(self._providers.values())

    def state(self, agent_id: str) -> RuntimeState:
        return self._state[agent_id]

    def unblock(self, agent_id: str) -> bool:
        """Lift a persisted cost/billing block. Paid agents stay blocked by the guard."""
        provider = self._providers[agent_id]
        if not self.cost_guard.allowed(provider.info)[0]:
            return False
        state = self._state[agent_id]
        lifted = self.store.unblock(agent_id) if self.store else False
        if state.health is Health.BLOCKED:
            state.health = Health.UNKNOWN
            state.blocked_reason = None
            state.cooldown_until = None
            state.last_error = None
            state.checked_at = None
            lifted = True
        return lifted

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
            elif report.health is Health.BLOCKED:
                # Re-detected on every start, so not persisted (unlike call-time blocks).
                state.health = Health.BLOCKED
                state.last_error = report.detail or "blocked by health check"
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
            self._expire_cooldown(state, now)
            if state.health is not Health.AVAILABLE:
                continue
            if self._minute_quota_full(provider, state, now):
                continue
            if not all(provider.supports(cap) for cap in required):
                continue
            result.append(provider)
        return result

    def _minute_quota_full(self, provider: AIProvider, state: RuntimeState, now: float) -> bool:
        limit = provider.info.rpm_limit
        while state.minute_window and now - state.minute_window[0] >= MINUTE:
            state.minute_window.popleft()
        if limit is None:
            return False
        window = self._group_window(provider.id, now)
        if len(window) < limit:
            return False
        self._cooldown(
            provider.id, window[0] + MINUTE - now, f"limite de {limit} pedidos/minuto (local)"
        )
        return True

    # -- learning from calls -------------------------------------------------

    def _count_call(self, agent_id: str, *, ok: bool) -> RuntimeState:
        state = self._state[agent_id]
        now = self._clock()
        self._roll_day(state, now)
        state.requests_today += 1
        state.minute_window.append(now)
        state.outcomes.append(ok)
        if self.store is not None:
            self.store.record_call(agent_id, state.day, ok=ok)
        return state

    def record_success(self, agent_id: str, response: AIResponse) -> None:
        state = self._count_call(agent_id, ok=True)
        now = self._clock()
        state.successes += 1
        state.consecutive_failures = 0
        state.last_error = None
        if state.avg_latency_ms is None:
            state.avg_latency_ms = response.latency_ms
        else:
            state.avg_latency_ms = (
                LATENCY_ALPHA * response.latency_ms + (1 - LATENCY_ALPHA) * state.avg_latency_ms
            )

        quota = response.rate_limit
        if quota is not None and quota.remaining is not None:
            for member in self._group(agent_id):
                self._state[member].provider_quota = quota
            if quota.remaining <= 0:
                wait = quota.reset_seconds or DEFAULT_RATE_LIMIT_COOLDOWN
                for member in self._group(agent_id):
                    self._cooldown(member, wait, "o provedor informou cota esgotada")
                return
        limit = self._providers[agent_id].info.daily_limit
        if limit is not None and self._group_requests_today(agent_id) >= limit:
            wait = self._seconds_until_utc_midnight(now)
            for member in self._group(agent_id):
                self._cooldown(member, wait, "cota diária usada")
            return
        self._persist(agent_id)

    def record_failure(self, agent_id: str, error: ProviderError) -> None:
        state = self._count_call(agent_id, ok=False)
        state.failures += 1
        state.consecutive_failures += 1
        state.last_error = str(error)

        if isinstance(error, (PaymentRequiredError, CostViolationError)):
            self._block(agent_id, f"{type(error).__name__}: {error}")
        elif isinstance(error, NotConfiguredError):
            state.health = Health.UNCONFIGURED
            self._persist(agent_id)
        elif isinstance(error, RateLimitError):
            wait = error.retry_after or DEFAULT_RATE_LIMIT_COOLDOWN
            # A quota shared by a group is exhausted for every member.
            members = (
                self._group(agent_id) if self._providers[agent_id].info.quota_group else [agent_id]
            )
            for member in members:
                self._cooldown(member, wait, str(error))
        else:
            delay = min(BACKOFF_BASE * 2 ** (state.consecutive_failures - 1), BACKOFF_MAX)
            if error.retry_after:
                delay = max(delay, error.retry_after)
            self._cooldown(agent_id, delay, str(error))

    # -- state transitions ------------------------------------------------------

    def _block(self, agent_id: str, reason: str) -> None:
        state = self._state[agent_id]
        state.health = Health.BLOCKED
        state.blocked_reason = reason
        state.cooldown_until = None
        state.last_error = reason
        self._persist(agent_id)

    def _cooldown(self, agent_id: str, seconds: float, reason: str) -> None:
        state = self._state[agent_id]
        state.health = Health.COOLDOWN
        state.cooldown_until = self._clock() + max(seconds, 0.0)
        state.last_error = reason
        self._persist(agent_id)

    def _persist(self, agent_id: str) -> None:
        if self.store is None:
            return
        state = self._state[agent_id]
        self.store.save_state(
            agent_id,
            now=self._clock(),
            cooldown_until=state.cooldown_until,
            blocked_reason=state.blocked_reason,
            last_error=state.last_error,
        )

    def _expire_cooldown(self, state: RuntimeState, now: float) -> None:
        if state.health is Health.COOLDOWN and not self._in_cooldown(state, now):
            state.health = Health.AVAILABLE

    def _in_cooldown(self, state: RuntimeState, now: float) -> bool:
        return state.cooldown_until is not None and now < state.cooldown_until

    def _roll_day(self, state: RuntimeState, now: float) -> None:
        day = utc_day(now)
        if state.day != day:
            state.day = day
            state.requests_today = 0
            state.provider_quota = None

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
        self._roll_day(state, now)
        self._expire_cooldown(state, now)
        while state.minute_window and now - state.minute_window[0] >= MINUTE:
            state.minute_window.popleft()

        remaining, source, limit = None, None, info.daily_limit
        quota = state.provider_quota
        if quota is not None and quota.remaining is not None:
            remaining, source, limit = quota.remaining, "provider", quota.limit or limit
        elif info.daily_limit is not None:
            used = self._group_requests_today(agent_id)
            remaining, source = max(info.daily_limit - used, 0), "local"

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
            remaining_source=source,
            rate_limit=limit,
            rpm_limit=info.rpm_limit,
            requests_last_minute=len(self._group_window(agent_id, now)),
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
            success_rate=state.success_rate,
            requests_today=state.requests_today,
            context_window=provider.context_window,
        )

    def statuses(self) -> list[AgentStatus]:
        return [self.status(agent_id) for agent_id in self._providers]

    async def close(self) -> None:
        await asyncio.gather(*(p.close() for p in self._providers.values()), return_exceptions=True)
        if self.store is not None:
            self.store.close()
