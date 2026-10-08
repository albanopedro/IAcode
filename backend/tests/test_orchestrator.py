import pytest

from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import (
    AllAgentsFailedError,
    PaymentRequiredError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.router import Router
from jarvis.core.types import Capability, CostClass, Health, Privacy, TaskType
from tests.fakes import FakeAgent, FakeClock


def setup(*agents):
    manager = AgentManager(agents, CostGuard(), clock=FakeClock())
    return manager, Orchestrator(manager)


async def test_answers_with_the_best_agent():
    manager, orch = setup(FakeAgent("low", priority=10), FakeAgent("high", priority=90))
    result = await orch.ask(Conversation(), "Olá JARVIS")
    assert result.response.agent_id == "high"
    assert [a.agent_id for a in result.attempts] == ["high"]
    assert manager.status("high").successes == 1


async def test_falls_back_when_the_first_agent_hits_its_limit():
    a = FakeAgent("a", [RateLimitError("429", retry_after=60)], priority=90)
    b = FakeAgent("b", [ProviderUnavailableError("down")], priority=80)
    c = FakeAgent("c", priority=70)
    manager, orch = setup(a, b, c)
    result = await orch.ask(Conversation(), "oi")
    assert result.response.agent_id == "c"
    assert [(x.agent_id, x.ok) for x in result.attempts] == [
        ("a", False),
        ("b", False),
        ("c", True),
    ]
    assert manager.status("a").health is Health.COOLDOWN
    assert manager.status("b").health is Health.COOLDOWN


async def test_agent_in_cooldown_is_skipped_on_the_next_message():
    a = FakeAgent("a", [RateLimitError("429", retry_after=60), "a voltou"], priority=90)
    b = FakeAgent("b", priority=10)
    _, orch = setup(a, b)
    conversation = Conversation()
    await orch.ask(conversation, "primeira")
    result = await orch.ask(conversation, "segunda")
    assert result.response.agent_id == "b"
    assert len(a.calls) == 1


async def test_capability_beats_raw_priority_for_the_task():
    chat_only = FakeAgent("chat", priority=70, capabilities={Capability.CHAT})
    coder = FakeAgent("coder", priority=50, capabilities={Capability.CHAT, Capability.CODE})
    _, orch = setup(chat_only, coder)
    result = await orch.ask(Conversation(), "Escreva um código Python que ordena uma lista")
    assert result.task is TaskType.CODE
    assert result.response.agent_id == "coder"
    # The chat-only agent was available, it just was not the best choice.
    assert chat_only.calls == []


async def test_context_is_kept_across_turns_and_agents():
    a = FakeAgent("a", ["Docker é uma plataforma...", RateLimitError("limite")], priority=90)
    b = FakeAgent("b", ["Use o Docker Desktop no Mac."], priority=10)
    _, orch = setup(a, b)
    conversation = Conversation()
    await orch.ask(conversation, "JARVIS, explique Docker.")
    result = await orch.ask(conversation, "E como eu instalaria isso no Mac?")
    assert result.response.agent_id == "b"
    sent = [(m.role, m.content) for m in b.calls[0].messages]
    assert sent[0][0] == "system"
    assert sent[1:] == [
        ("user", "JARVIS, explique Docker."),
        ("assistant", "Docker é uma plataforma..."),
        ("user", "E como eu instalaria isso no Mac?"),
    ]
    assert [t.agent_id for t in conversation.turns] == [None, "a", None, "b"]


async def test_reported_cost_blocks_the_agent_and_falls_back():
    sneaky = FakeAgent("sneaky", cost=0.02, priority=90)
    honest = FakeAgent("honest", priority=10)
    manager, orch = setup(sneaky, honest)
    conversation = Conversation()
    result = await orch.ask(conversation, "oi")
    assert result.response.agent_id == "honest"
    assert manager.status("sneaky").health is Health.BLOCKED
    assert "custo" not in result.response.text
    # The paid answer is discarded, never shown or stored.
    assert all(t.message.content != "resposta de sneaky" for t in conversation.turns)


async def test_payment_required_blocks_the_agent():
    manager, orch = setup(
        FakeAgent("a", [PaymentRequiredError("402")], priority=90), FakeAgent("b")
    )
    await orch.ask(Conversation(), "oi")
    assert manager.status("a").health is Health.BLOCKED


async def test_paid_agent_is_never_called():
    paid = FakeAgent("paid", cost_class=CostClass.PAID, priority=100, allow_paid=True)
    free = FakeAgent("free", priority=1)
    _, orch = setup(paid, free)
    result = await orch.ask(Conversation(), "oi")
    assert result.response.agent_id == "free"
    assert paid.calls == []


async def test_all_agents_failing_raises_and_forgets_the_message():
    _, orch = setup(
        FakeAgent("a", [ProviderUnavailableError("x")]),
        FakeAgent("b", [RateLimitError("y")]),
    )
    conversation = Conversation()
    with pytest.raises(AllAgentsFailedError) as info:
        await orch.ask(conversation, "oi")
    assert len(info.value.attempts) == 2
    assert conversation.turns == []


async def test_no_available_agent_gives_a_clear_error():
    _, orch = setup(FakeAgent("a", health=Health.OFFLINE))
    with pytest.raises(AllAgentsFailedError, match="no free agent"):
        await orch.ask(Conversation(), "oi")


async def test_empty_message_is_rejected():
    _, orch = setup(FakeAgent("a"))
    with pytest.raises(ValueError):
        await orch.ask(Conversation(), "   ")


async def test_router_penalizes_failures_and_prefers_privacy():
    manager = AgentManager(
        [
            FakeAgent("private", priority=50, privacy=Privacy.ZERO_RETENTION),
            FakeAgent("plain", priority=50),
        ],
        CostGuard(),
        clock=FakeClock(),
    )
    router = Router(manager)
    ranked = router.rank(manager.providers, TaskType.CHAT)
    assert ranked[0].id == "private"
    manager.state("private").consecutive_failures = 1
    ranked = router.rank(manager.providers, TaskType.CHAT)
    assert ranked[0].id == "plain"
