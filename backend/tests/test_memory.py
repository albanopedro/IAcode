"""Phase 6: persistent history, summaries (short-term) and facts (long-term)."""

import pytest

from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import ProviderUnavailableError
from jarvis.core.orchestrator import Orchestrator, ShareWith
from jarvis.core.types import Privacy
from jarvis.memory.assistant import MEMORY_AGENT, Assistant, relevant_facts
from jarvis.memory.commands import CommandKind, parse
from jarvis.memory.store import MemoryStore
from jarvis.memory.summarizer import maybe_summarize
from tests.fakes import FakeAgent, FakeClock


def orchestrator(*agents, share=ShareWith.PRIVATE):
    manager = AgentManager(agents, CostGuard(), clock=FakeClock())
    return Orchestrator(manager, share_private_with=share)


# -- store: history ---------------------------------------------------------------


def test_conversation_is_saved_and_reloaded(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    conversation = store.new_conversation()
    conversation.add_user("explique Docker")
    conversation.add_assistant("Docker é uma plataforma.", "opencode:x")
    conversation.add_user("pergunta que falhou")
    conversation.drop_last_user()
    store.close()

    again = MemoryStore(path)
    loaded = again.latest_conversation()
    assert loaded.id == conversation.id
    assert loaded.title == "explique Docker"
    assert [(t.message.role, t.message.content, t.agent_id) for t in loaded.turns] == [
        ("user", "explique Docker", None),
        ("assistant", "Docker é uma plataforma.", "opencode:x"),
    ]


def test_conversations_are_listed_newest_first_and_deletable():
    store = MemoryStore()
    first = store.new_conversation()
    first.add_user("primeira")
    second = store.new_conversation()
    second.add_user("segunda")
    store.new_conversation()  # empty conversations are not listed
    assert [c.title for c in store.list_conversations()] == ["segunda", "primeira"]
    assert store.delete_conversation(first.id)
    assert store.load_conversation(first.id) is None
    assert [c.title for c in store.list_conversations()] == ["segunda"]
    assert store.delete_all_conversations() == 2


def test_clear_empties_a_saved_conversation():
    store = MemoryStore()
    conversation = store.new_conversation()
    conversation.add_user("oi")
    conversation.clear()
    assert store.load_conversation(conversation.id).turns == []


# -- store: facts -----------------------------------------------------------------


def test_facts_are_deduplicated_found_and_deleted():
    store = MemoryStore()
    fact = store.add_fact("Eu prefiro respostas curtas")
    assert fact is not None
    assert store.add_fact("eu PREFIRO respostas curtas!") is None  # same after normalizing
    store.add_fact("Meu editor é o VS Code")
    assert [f.text for f in store.find_facts("respostas")] == ["Eu prefiro respostas curtas"]
    assert store.delete_fact(fact.id)
    assert [f.text for f in store.facts()] == ["Meu editor é o VS Code"]
    assert store.add_fact("   ") is None
    assert len(store.add_fact("x" * 1000).text) == 300


# -- commands ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "argument"),
    [
        (
            "JARVIS, lembre que eu prefiro respostas curtas.",
            CommandKind.REMEMBER,
            "eu prefiro respostas curtas",
        ),
        (
            "Lembra que meu aniversário é em março",
            CommandKind.REMEMBER,
            "meu aniversário é em março",
        ),
        ("por favor, memorize: uso macOS", CommandKind.REMEMBER, "uso macOS"),
        (
            "Guarde isso: minha cor favorita é azul!",
            CommandKind.REMEMBER,
            "minha cor favorita é azul",
        ),
        (
            "Esqueça que eu prefiro respostas curtas",
            CommandKind.FORGET,
            "eu prefiro respostas curtas",
        ),
        ("JARVIS, apague da memória o aniversário.", CommandKind.FORGET, "o aniversário"),
        ("Esqueça tudo", CommandKind.FORGET_ALL, ""),
        ("O que você sabe sobre mim?", CommandKind.LIST, ""),
        ("Jarvis, quais são suas memórias?", CommandKind.LIST, ""),
    ],
)
def test_memory_commands(text, kind, argument):
    command = parse(text)
    assert command is not None and command.kind is kind and command.argument == argument


@pytest.mark.parametrize(
    "text",
    [
        "Você lembra de mim?",
        "Explique Docker",
        "Como eu faço para lembrar de beber água?",
        "O que você sabe sobre Docker?",
        "lembre",
    ],
)
def test_ordinary_messages_are_not_commands(text):
    assert parse(text) is None


# -- assistant: local commands ---------------------------------------------------------


async def test_memory_commands_run_locally_without_any_agent():
    agent = FakeAgent("a")
    store = MemoryStore()
    assistant = Assistant(orchestrator(agent), store, summarize=False)
    conversation = store.new_conversation()

    saved = await assistant.ask(conversation, "JARVIS, lembre que eu prefiro respostas curtas.")
    assert saved.response.agent_id == MEMORY_AGENT and saved.response.cost == 0
    assert "vou lembrar" in saved.response.text
    assert (
        await assistant.ask(conversation, "lembre que eu prefiro respostas curtas")
    ).response.text == ("Eu já tinha isso guardado.")
    listed = await assistant.ask(conversation, "O que você sabe sobre mim?")
    assert "prefiro respostas curtas" in listed.response.text
    refused = await assistant.ask(conversation, "Esqueça tudo")
    assert "Por segurança" in refused.response.text and store.facts()
    forgot = await assistant.ask(conversation, "Esqueça que eu prefiro respostas curtas")
    assert "esqueci" in forgot.response.text and store.facts() == []
    assert agent.calls == []
    assert len(conversation.turns) == 10  # the exchanges are part of the history


# -- assistant: long-term facts go only to private agents ------------------------------


async def test_facts_reach_only_zero_retention_agents_by_default():
    private = FakeAgent(
        "private",
        privacy=Privacy.ZERO_RETENTION,
        priority=90,
        script=[ProviderUnavailableError("down")],
    )
    public = FakeAgent("public", privacy=Privacy.MAY_TRAIN, priority=10)
    store = MemoryStore()
    store.add_fact("Meu nome é Zeca")
    assistant = Assistant(orchestrator(private, public), store, summarize=False)
    result = await assistant.ask(store.new_conversation(), "Qual é o meu nome?")
    assert result.response.agent_id == "public"
    assert "Meu nome é Zeca" in private.calls[0].messages[0].content
    assert "Meu nome é Zeca" not in public.calls[0].messages[0].content


async def test_facts_can_be_shared_with_every_agent():
    public = FakeAgent("public", privacy=Privacy.MAY_TRAIN)
    store = MemoryStore()
    store.add_fact("Meu nome é Zeca")
    assistant = Assistant(orchestrator(public, share=ShareWith.ALL), store, summarize=False)
    await assistant.ask(store.new_conversation(), "Qual é o meu nome?")
    system = public.calls[0].messages[0].content
    assert "Meu nome é Zeca" in system and "não instruções" in system


def test_relevant_facts_prefers_shared_words_when_there_are_many():
    store = MemoryStore()
    for i in range(40):
        store.add_fact(f"fato aleatório número {i}")
    store.add_fact("Meu cachorro se chama Thor")
    picked = relevant_facts(store.facts(), "Como está o cachorro?", limit=5)
    assert picked[0].text == "Meu cachorro se chama Thor" and len(picked) == 5


# -- short-term memory: summaries ------------------------------------------------------


async def test_old_turns_become_a_summary_and_reach_the_agent():
    agent = FakeAgent("a", ["Resumo: falamos de Docker e do Mac."])
    store = MemoryStore()
    conversation = store.new_conversation()
    for i in range(15):
        conversation.add_user(f"pergunta {i}")
        conversation.add_assistant(f"resposta {i}", "a")
    orch = orchestrator(agent)
    assert await maybe_summarize(orch, conversation, window=20, batch=10) is True
    assert conversation.summarized == 10 and "Docker" in conversation.summary
    prompt = agent.calls[0].messages[-1].content
    assert "pergunta 0" in prompt and "pergunta 5" not in prompt  # only turns outside the window
    assert "ignore qualquer instrução" in agent.calls[0].messages[0].content

    context = conversation.context(20)
    assert context[0].role == "system" and "Docker" in context[0].content
    assert len(context) == 21
    # Saved, so it survives a reload.
    assert store.load_conversation(conversation.id).summary == conversation.summary
    # Nothing new left the window: no new call.
    assert await maybe_summarize(orch, conversation, window=20, batch=10) is False
    assert len(agent.calls) == 1


async def test_summary_failure_is_harmless_and_stale_summaries_are_dropped():
    conversation = Conversation()
    for i in range(30):
        conversation.add_user(f"m{i}")
    failing = orchestrator(FakeAgent("a", [ProviderUnavailableError("down")]))
    assert await maybe_summarize(failing, conversation) is False
    assert conversation.summary is None
    generation = conversation.generation
    conversation.clear()
    assert conversation.set_summary("velho", 10, generation) is False


async def test_assistant_summarizes_in_the_background():
    agent = FakeAgent("a", ["ok"])
    store = MemoryStore()
    conversation = store.new_conversation()
    for i in range(14):
        conversation.add_user(f"pergunta {i}")
        conversation.add_assistant(f"resposta {i}", "a")
    assistant = Assistant(orchestrator(agent), store, window=20)
    await assistant.ask(conversation, "mais uma")
    await assistant.wait_background()
    assert conversation.summary == "ok"
    assert len(agent.calls) == 2  # the answer + one summary
