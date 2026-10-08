"""Phase 3: persisted usage, per-minute limits, provider quotas and quota groups."""

import pytest

from jarvis.core.agent_manager import MINUTE, AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import PaymentRequiredError, ProviderUnavailableError, RateLimitError
from jarvis.core.types import AIResponse, CostClass, Health, RateLimitInfo
from jarvis.core.usage_store import UsageStore
from tests.fakes import FakeAgent, FakeClock


def make(*agents, clock=None, store=None):
    return AgentManager(agents, CostGuard(), clock=clock or FakeClock(), store=store)


def ok(agent_id, rate_limit=None):
    return AIResponse(text="ok", agent_id=agent_id, model="m", rate_limit=rate_limit)


async def ready(manager):
    await manager.refresh_health(force=True)
    return manager


# -- persistence -------------------------------------------------------------------


async def test_daily_usage_survives_a_restart(tmp_path):
    clock, path = FakeClock(), tmp_path / "jarvis.db"
    first = await ready(make(FakeAgent("a", daily_limit=3), clock=clock, store=UsageStore(path)))
    first.record_success("a", ok("a"))
    first.record_success("a", ok("a"))
    await first.close()

    second = await ready(make(FakeAgent("a", daily_limit=3), clock=clock, store=UsageStore(path)))
    status = second.status("a")
    assert status.requests_today == 2
    assert (status.remaining_usage, status.remaining_source) == (1, "local")


async def test_cooldown_survives_a_restart(tmp_path):
    clock, path = FakeClock(), tmp_path / "jarvis.db"
    first = await ready(make(FakeAgent("a"), clock=clock, store=UsageStore(path)))
    first.record_failure("a", RateLimitError("429", retry_after=300))
    await first.close()

    clock.advance(10)
    second = await ready(make(FakeAgent("a"), clock=clock, store=UsageStore(path)))
    assert second.status("a").health is Health.COOLDOWN
    clock.advance(300)
    assert [p.id for p in second.candidates()] == ["a"]


async def test_billing_block_survives_restarts_until_unblocked(tmp_path):
    clock, path = FakeClock(), tmp_path / "jarvis.db"
    first = await ready(make(FakeAgent("a"), clock=clock, store=UsageStore(path)))
    first.record_failure("a", PaymentRequiredError("HTTP 402"))
    await first.close()

    clock.advance(7 * 24 * 3600)  # a week later, still blocked
    second = await ready(make(FakeAgent("a"), clock=clock, store=UsageStore(path)))
    status = second.status("a")
    assert status.health is Health.BLOCKED
    assert "jarvis unblock a" in status.last_error
    assert second.candidates() == []

    assert second.unblock("a") is True
    await second.refresh_health(force=True)
    assert [p.id for p in second.candidates()] == ["a"]
    await second.close()

    third = await ready(make(FakeAgent("a"), clock=clock, store=UsageStore(path)))
    assert third.status("a").health is Health.AVAILABLE


async def test_paid_agents_cannot_be_unblocked():
    manager = await ready(make(FakeAgent("paid", cost_class=CostClass.PAID)))
    assert manager.unblock("paid") is False
    assert manager.status("paid").health is Health.BLOCKED


async def test_health_check_blocks_are_not_persisted(tmp_path):
    path = tmp_path / "jarvis.db"
    manager = await ready(make(FakeAgent("a", health=Health.BLOCKED), store=UsageStore(path)))
    assert manager.status("a").health is Health.BLOCKED
    await manager.close()
    assert UsageStore(path).blocked() == {}


# -- per-minute limit ------------------------------------------------------------


async def test_rpm_limit_is_respected_before_the_provider_complains():
    clock = FakeClock()
    manager = await ready(make(FakeAgent("a", rpm_limit=2), clock=clock))
    manager.record_success("a", ok("a"))
    clock.advance(10)
    manager.record_success("a", ok("a"))
    assert manager.candidates() == []
    status = manager.status("a")
    assert status.health is Health.COOLDOWN
    assert status.requests_last_minute == 2
    clock.advance(MINUTE - 10)  # the first call leaves the window
    assert [p.id for p in manager.candidates()] == ["a"]


# -- provider-reported quota -----------------------------------------------------


async def test_provider_quota_is_shown_and_exhaustion_cools_down():
    clock = FakeClock()
    manager = await ready(make(FakeAgent("a", daily_limit=1000), clock=clock))
    manager.record_success("a", ok("a", RateLimitInfo(limit=1000, remaining=420)))
    status = manager.status("a")
    assert (status.remaining_usage, status.remaining_source, status.rate_limit) == (
        420,
        "provider",
        1000,
    )
    manager.record_success("a", ok("a", RateLimitInfo(remaining=0, reset_seconds=90)))
    assert manager.status("a").health is Health.COOLDOWN
    clock.advance(91)
    assert [p.id for p in manager.candidates()] == ["a"]


# -- quota groups ----------------------------------------------------------------


async def test_quota_group_shares_daily_and_minute_counters():
    clock = FakeClock()
    a = FakeAgent("or:a", daily_limit=3, rpm_limit=10, quota_group="openrouter")
    b = FakeAgent("or:b", daily_limit=3, rpm_limit=10, quota_group="openrouter")
    solo = FakeAgent("solo", daily_limit=3)
    manager = await ready(make(a, b, solo, clock=clock))
    manager.record_success("or:a", ok("or:a"))
    manager.record_success("or:b", ok("or:b"))
    assert manager.status("or:a").remaining_usage == 1
    assert manager.status("or:b").requests_last_minute == 2
    manager.record_success("or:a", ok("or:a"))  # 3rd call of the group: quota used
    assert manager.status("or:a").health is Health.COOLDOWN
    assert manager.status("or:b").health is Health.COOLDOWN
    assert manager.status("solo").health is Health.AVAILABLE


async def test_rate_limit_on_one_member_cools_down_the_whole_group():
    a = FakeAgent("or:a", quota_group="openrouter")
    b = FakeAgent("or:b", quota_group="openrouter")
    other = FakeAgent("groq:x")
    manager = await ready(make(a, b, other))
    manager.record_failure("or:a", RateLimitError("daily limit", retry_after=600))
    assert [p.id for p in manager.candidates()] == ["groq:x"]


async def test_plain_failure_only_affects_the_failing_member():
    a = FakeAgent("or:a", quota_group="openrouter")
    b = FakeAgent("or:b", quota_group="openrouter")
    manager = await ready(make(a, b))
    manager.record_failure("or:a", ProviderUnavailableError("model down"))
    assert [p.id for p in manager.candidates()] == ["or:b"]


# -- success rate ----------------------------------------------------------------


async def test_success_rate_over_recent_calls():
    manager = await ready(make(FakeAgent("a")))
    manager.record_success("a", ok("a"))
    manager.record_failure("a", ProviderUnavailableError("x"))
    manager.record_success("a", ok("a"))
    manager.record_success("a", ok("a"))
    assert manager.status("a").success_rate == pytest.approx(0.75)
