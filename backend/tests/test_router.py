from jarvis.core.agent_manager import AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import ProviderUnavailableError
from jarvis.core.router import OUTPUT_RESERVE, Router, estimate_tokens
from jarvis.core.types import AIRequest, AIResponse, Capability, Message, TaskType
from tests.fakes import FakeAgent, FakeClock


def setup(*agents):
    manager = AgentManager(agents, CostGuard(), clock=FakeClock())
    return manager, Router(manager)


def test_configured_quality_beats_priority_for_its_task():
    generalist = FakeAgent("generalist", priority=70, capabilities={Capability.CHAT})
    math_whiz = FakeAgent("math", priority=50, quality={TaskType.MATH: 0.95})
    _, router = setup(generalist, math_whiz)
    assert router.rank([generalist, math_whiz], TaskType.MATH)[0].id == "math"
    assert router.rank([generalist, math_whiz], TaskType.CHAT)[0].id == "generalist"


def test_quality_falls_back_to_capabilities_and_is_clamped():
    coder = FakeAgent("c", capabilities={Capability.CHAT, Capability.CODE})
    weird = FakeAgent("w", quality={TaskType.CHAT: 7})
    assert Router.quality(coder, TaskType.CODE) == 0.6
    assert Router.quality(coder, TaskType.MATH) == 0.2
    assert Router.quality(weird, TaskType.CHAT) == 1.0


def test_low_success_rate_lowers_the_score():
    flaky, steady = FakeAgent("flaky"), FakeAgent("steady")
    manager, router = setup(flaky, steady)
    for _ in range(3):
        manager.record_success("flaky", AIResponse(text="x", agent_id="flaky", model="m"))
        manager.record_failure("flaky", ProviderUnavailableError("x"))
    manager.record_success("flaky", AIResponse(text="x", agent_id="flaky", model="m"))
    manager.record_success("steady", AIResponse(text="x", agent_id="steady", model="m"))
    parts = router.explain(flaky, TaskType.CHAT)
    assert parts["success"] < 0
    assert router.rank([flaky, steady], TaskType.CHAT)[0].id == "steady"


def test_agents_with_a_too_small_context_window_are_filtered():
    tiny = FakeAgent("tiny", priority=99, context_window=2000)
    big = FakeAgent("big", priority=1, context_window=200_000)
    unknown = FakeAgent("unknown", priority=1)
    _, router = setup(tiny, big, unknown)
    long_request = AIRequest(messages=[Message(role="user", content="x" * 20_000)])
    ranked = router.rank([tiny, big, unknown], TaskType.CHAT, long_request)
    assert [p.id for p in ranked] == ["big", "unknown"]
    short = AIRequest(messages=[Message(role="user", content="oi")])
    assert router.rank([tiny, big], TaskType.CHAT, short)[0].id == "tiny"


def test_estimate_tokens_keeps_room_for_the_answer():
    request = AIRequest(messages=[Message(role="user", content="a" * 350)])
    assert estimate_tokens(request) == 100 + OUTPUT_RESERVE


def test_explain_adds_up_to_the_score():
    agent = FakeAgent("a", priority=42)
    _, router = setup(agent)
    assert sum(router.explain(agent, TaskType.CHAT).values()) == router.score(agent, TaskType.CHAT)
