import pytest

from jarvis.core.agent_manager import BACKOFF_BASE, BACKOFF_MAX, AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import (
    CostViolationError,
    NotConfiguredError,
    PaymentRequiredError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.core.types import AIResponse, Capability, CostClass, Health
from tests.fakes import FakeAgent, FakeClock


def make(*agents, clock=None):
    return AgentManager(agents, CostGuard(), clock=clock or FakeClock())


def ok_response(agent_id, latency=100.0):
    return AIResponse(text="ok", agent_id=agent_id, model="m", latency_ms=latency)


async def test_refresh_marks_agents_by_health():
    manager = make(
        FakeAgent("up"),
        FakeAgent("down", health=Health.OFFLINE),
        FakeAgent("nokey", health=Health.UNCONFIGURED),
    )
    await manager.refresh_health()
    assert [p.id for p in manager.candidates()] == ["up"]
    assert manager.status("down").health is Health.OFFLINE
    assert manager.status("nokey").health is Health.UNCONFIGURED


async def test_paid_agent_is_blocked_at_registration_and_never_checked():
    paid = FakeAgent("paid", cost_class=CostClass.PAID)
    manager = make(paid)
    await manager.refresh_health(force=True)
    assert manager.status("paid").health is Health.BLOCKED
    assert manager.status("paid").requires_payment is True
    assert paid.health_checks == 0
    assert manager.candidates() == []


async def test_health_is_cached_until_ttl():
    clock = FakeClock()
    agent = FakeAgent("a")
    manager = make(agent, clock=clock)
    await manager.refresh_health()
    await manager.refresh_health()
    assert agent.health_checks == 1
    clock.advance(61)
    await manager.refresh_health()
    assert agent.health_checks == 2


async def test_rate_limit_uses_retry_after_then_recovers():
    clock = FakeClock()
    manager = make(FakeAgent("a"), clock=clock)
    await manager.refresh_health()
    manager.record_failure("a", RateLimitError("429", retry_after=30))
    status = manager.status("a")
    assert status.health is Health.COOLDOWN and status.available is False
    assert manager.candidates() == []
    clock.advance(31)
    assert [p.id for p in manager.candidates()] == ["a"]


async def test_healthy_check_does_not_cancel_cooldown():
    clock = FakeClock()
    manager = make(FakeAgent("a"), clock=clock)
    await manager.refresh_health()
    manager.record_failure("a", RateLimitError("429", retry_after=300))
    await manager.refresh_health(force=True)
    assert manager.status("a").health is Health.COOLDOWN


async def test_repeated_failures_back_off_exponentially_up_to_a_cap():
    clock = FakeClock()
    manager = make(FakeAgent("a"), clock=clock)
    await manager.refresh_health()
    expected = [BACKOFF_BASE * 2**i for i in range(3)]
    for delay in expected:
        manager.record_failure("a", ProviderUnavailableError("down"))
        state = manager.state("a")
        assert state.cooldown_until == pytest.approx(clock.now + delay)
        clock.advance(delay + 1)
    for _ in range(10):
        manager.record_failure("a", ProviderUnavailableError("down"))
    assert manager.state("a").cooldown_until - clock.now == BACKOFF_MAX


@pytest.mark.parametrize("error", [PaymentRequiredError("402"), CostViolationError("cost")])
async def test_billing_problems_block_for_good(error):
    clock = FakeClock()
    manager = make(FakeAgent("a"), clock=clock)
    await manager.refresh_health()
    manager.record_failure("a", error)
    clock.advance(10_000)
    await manager.refresh_health(force=True)
    assert manager.status("a").health is Health.BLOCKED
    assert manager.candidates() == []


async def test_not_configured_error_marks_unconfigured():
    manager = make(FakeAgent("a"))
    await manager.refresh_health()
    manager.record_failure("a", NotConfiguredError("no key"))
    assert manager.status("a").health is Health.UNCONFIGURED


async def test_success_resets_failures_and_tracks_latency():
    manager = make(FakeAgent("a"))
    await manager.refresh_health()
    manager.record_failure("a", RateLimitError("429", retry_after=1))
    manager.record_success("a", ok_response("a", 1000))
    manager.record_success("a", ok_response("a", 2000))
    status = manager.status("a")
    assert status.consecutive_failures == 0
    assert status.successes == 2 and status.failures == 1
    assert status.avg_latency_ms == pytest.approx(0.3 * 2000 + 0.7 * 1000)


async def test_daily_quota_triggers_cooldown_until_midnight_utc():
    clock = FakeClock(1_800_000_000.0)  # 2027-01-15 08:00 UTC
    manager = make(FakeAgent("a", daily_limit=2), clock=clock)
    await manager.refresh_health()
    manager.record_success("a", ok_response("a"))
    assert manager.status("a").remaining_usage == 1
    manager.record_success("a", ok_response("a"))
    status = manager.status("a")
    assert status.remaining_usage == 0
    assert status.health is Health.COOLDOWN
    assert status.cooldown_until.hour == 0 and status.cooldown_until.minute == 0
    clock.advance(16 * 3600 + 1)  # past midnight: counter resets
    assert [p.id for p in manager.candidates()] == ["a"]
    assert manager.status("a").requests_today == 0


async def test_candidates_respect_hard_requirements():
    manager = make(FakeAgent("text"), FakeAgent("eyes", capabilities={Capability.VISION}))
    await manager.refresh_health()
    ids = [p.id for p in manager.candidates(frozenset({Capability.VISION}))]
    assert ids == ["eyes"]


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make(FakeAgent("a"), FakeAgent("a"))


async def test_crashing_health_check_counts_as_offline():
    class Broken(FakeAgent):
        async def check_health(self):
            raise RuntimeError("boom")

    manager = make(Broken("b"))
    await manager.refresh_health()
    assert manager.status("b").health is Health.OFFLINE
    assert "boom" in manager.status("b").last_error
