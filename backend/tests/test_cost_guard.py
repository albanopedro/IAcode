import pytest

from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import CostViolationError
from jarvis.core.types import AIResponse, CostClass, CostMode
from tests.fakes import FakeAgent


def response(cost):
    return AIResponse(text="ok", agent_id="a", model="m", cost=cost)


@pytest.mark.parametrize(
    "cost_class", [CostClass.LOCAL, CostClass.FREE, CostClass.FREE_WITH_LIMITS]
)
def test_free_classes_are_allowed(cost_class):
    ok, _ = CostGuard().allowed(FakeAgent("a", cost_class=cost_class).info)
    assert ok


def test_paid_is_blocked_by_default():
    guard = CostGuard()
    info = FakeAgent("paid", cost_class=CostClass.PAID, allow_paid=True).info
    assert guard.mode is CostMode.FREE_ONLY
    assert guard.allowed(info)[0] is False
    with pytest.raises(CostViolationError):
        guard.check_before_call(info)


def test_paid_needs_both_global_mode_and_agent_opt_in():
    guard = CostGuard(CostMode.ALLOW_PAID)
    assert guard.allowed(FakeAgent("p", cost_class=CostClass.PAID).info)[0] is False
    opted_in = FakeAgent("p", cost_class=CostClass.PAID, allow_paid=True).info
    assert guard.allowed(opted_in)[0] is True


@pytest.mark.parametrize("cost", [None, 0, 0.0])
def test_zero_or_unknown_cost_passes(cost):
    CostGuard().check_after_call(FakeAgent("a").info, response(cost))


def test_any_reported_cost_is_a_violation():
    with pytest.raises(CostViolationError, match="0.0001"):
        CostGuard().check_after_call(FakeAgent("a").info, response(0.0001))
