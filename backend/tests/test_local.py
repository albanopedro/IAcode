"""Phase 8: the local (offline) model and the local-only mode."""

import json

import httpx
import pytest
import respx

from jarvis.agents.local.mlx import NO_ORIGIN, LocalMLXAgent, LocalModelStore, MLXServer
from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import AllAgentsFailedError, NotConfiguredError, ProviderUnavailableError
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.types import AIRequest, CostClass, Health, Message, Privacy
from jarvis.memory.assistant import Assistant
from jarvis.memory.store import MemoryStore
from jarvis.memory.summarizer import maybe_summarize
from tests.fakes import FakeAgent, FakeClock

REPO = "mlx-community/Tiny-Model-4bit"


class FakeServer(MLXServer):
    """Pretends to be running at a fixed URL; respx answers the HTTP calls."""

    def __init__(self):
        super().__init__(model_path=None)
        self.starts = 0

    async def ensure_started(self):
        self.starts += 1
        self.url = "http://127.0.0.1:9999"
        return self.url

    async def close(self):
        pass


def downloaded_store(tmp_path):
    store = LocalModelStore(tmp_path)
    path = store.path(REPO)
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"0" * 10)
    return store


# -- store -------------------------------------------------------------------------


def test_model_store_paths_and_download_state(tmp_path):
    store = LocalModelStore(tmp_path)
    assert store.path(REPO).name == "mlx-community--Tiny-Model-4bit"
    assert not store.is_downloaded(REPO)
    store = downloaded_store(tmp_path)
    assert store.is_downloaded(REPO) and store.size_bytes(REPO) > 0
    assert store.remove(REPO) and not store.is_downloaded(REPO)


# -- server command: locked down ----------------------------------------------------


def test_server_is_local_only_rejects_browsers_and_stays_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    server = MLXServer(tmp_path / "model")
    command = server.command(12345)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--allowed-origins") + 1] == NO_ORIGIN
    env = server.environment()
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert "OPENAI_API_KEY" not in env


# -- agent ---------------------------------------------------------------------------


async def test_agent_is_local_free_and_private(tmp_path):
    agent = LocalMLXAgent(REPO, LocalModelStore(tmp_path), server=FakeServer())
    info = agent.info
    assert (info.cost_class, info.is_local, info.privacy) == (CostClass.LOCAL, True, Privacy.LOCAL)
    assert info.id == "local:Tiny-Model-4bit"
    report = await agent.check_health()
    assert report.health is Health.UNCONFIGURED and "jarvis local download" in report.detail
    with pytest.raises(NotConfiguredError):
        await agent.generate(AIRequest(messages=[Message(role="user", content="oi")]))


@respx.mock
async def test_agent_answers_and_strips_thinking(tmp_path):
    server = FakeServer()
    agent = LocalMLXAgent(REPO, downloaded_store(tmp_path), server=server)
    assert (await agent.check_health()).health is Health.AVAILABLE
    assert server.starts == 0  # healthy without loading the model
    route = respx.post("http://127.0.0.1:9999/v1/chat/completions").respond(
        200,
        json={
            "choices": [{"message": {"content": "<think>hmm</think>\nOlá! Sou o JARVIS."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        },
    )
    response = await agent.generate(AIRequest(messages=[Message(role="user", content="oi")]))
    assert response.text == "Olá! Sou o JARVIS." and response.cost == 0
    assert (response.input_tokens, response.output_tokens) == (12, 5)
    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [{"role": "user", "content": "oi"}]


@respx.mock
async def test_agent_errors_become_provider_errors(tmp_path):
    agent = LocalMLXAgent(REPO, downloaded_store(tmp_path), server=FakeServer())
    request = AIRequest(messages=[Message(role="user", content="oi")])
    respx.post("http://127.0.0.1:9999/v1/chat/completions").respond(500)
    with pytest.raises(ProviderUnavailableError):
        await agent.generate(request)
    respx.post("http://127.0.0.1:9999/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(ProviderUnavailableError):
        await agent.generate(request)


# -- offline fallback and local-only mode ------------------------------------------------


def local_and_online(local_script=("resposta local",), online_script=("resposta online",)):
    local = FakeAgent("local", list(local_script), priority=30, privacy=Privacy.LOCAL)
    local.info = local.info.model_copy(update={"is_local": True, "cost_class": CostClass.LOCAL})
    online = FakeAgent("online", list(online_script), priority=70)
    manager = AgentManager([local, online], CostGuard(), clock=FakeClock())
    return local, online, Orchestrator(manager)


async def test_online_first_local_takes_over_when_offline():
    local, online, orch = local_and_online(online_script=[ProviderUnavailableError("sem internet")])
    result = await orch.ask(Conversation(), "oi")
    assert result.response.agent_id == "local"
    assert [a.agent_id for a in result.attempts] == ["online", "local"]


async def test_local_only_never_calls_online_agents():
    local, online, orch = local_and_online()
    store = MemoryStore()
    assistant = Assistant(orch, store, summarize=False)
    result = await assistant.ask(store.new_conversation(), "assunto sensível", local_only=True)
    assert result.response.agent_id == "local"
    assert online.calls == []


async def test_local_only_without_a_local_model_fails_clearly():
    online = FakeAgent("online")
    orch = Orchestrator(AgentManager([online], CostGuard(), clock=FakeClock()))
    with pytest.raises(AllAgentsFailedError, match="jarvis local download"):
        await orch.ask(Conversation(), "oi", local_only=True)
    assert online.calls == []


async def test_summaries_stay_local_in_local_only_mode():
    local, online, orch = local_and_online(local_script=["resumo local"])
    conversation = Conversation()
    for i in range(15):
        conversation.add_user(f"pergunta {i}")
        conversation.add_assistant(f"resposta {i}", "local")
    assert await maybe_summarize(orch, conversation, local_only=True)
    assert conversation.summary == "resumo local" and online.calls == []


async def test_local_agent_receives_private_facts():
    local, online, orch = local_and_online()
    store = MemoryStore()
    store.add_fact("Meu nome é Zeca")
    assistant = Assistant(orch, store, summarize=False)
    await assistant.ask(store.new_conversation(), "qual é meu nome?", local_only=True)
    assert "Meu nome é Zeca" in local.calls[0].messages[0].content


# -- private (local-only) turns ---------------------------------------------------------


def test_private_turns_persist_and_stay_out_of_non_local_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    conversation = store.new_conversation()
    conversation.add_user("meu segredo", private=True)
    conversation.add_assistant("guardado", "local", private=True)
    conversation.add_user("pergunta comum")
    store.close()

    loaded = MemoryStore(tmp_path / "memory.db").load_conversation(conversation.id)
    assert [t.private for t in loaded.turns] == [True, True, False]
    assert loaded.title == "🔒 conversa privada"  # the title never shows private text
    public = [m.content for m in loaded.context()]
    assert public == ["pergunta comum"]
    assert "meu segredo" in [m.content for m in loaded.context(include_private=True)]


def test_old_databases_get_the_private_column(tmp_path):
    import sqlite3

    path = tmp_path / "memory.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
        "created_at REAL NOT NULL, updated_at REAL NOT NULL, summary TEXT, "
        "summarized INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, "
        "role TEXT NOT NULL, content TEXT NOT NULL, agent_id TEXT, created_at REAL NOT NULL);"
        "INSERT INTO conversations VALUES ('c1', 'antiga', 1, 1, NULL, 0);"
        "INSERT INTO messages (conversation_id, role, content, created_at) "
        "VALUES ('c1', 'user', 'oi', 1);"
    )
    old.commit()
    old.close()
    loaded = MemoryStore(path).load_conversation("c1")
    assert loaded.turns[0].message.content == "oi" and loaded.turns[0].private is False


async def test_summaries_never_include_private_turns():
    local, online, orch = local_and_online(online_script=["resumo"])
    conversation = Conversation()
    for i in range(20):  # 40 turns: the 20 outside the window are pairs 0–9
        conversation.add_user(f"segredo {i}", private=i < 5)
        conversation.add_assistant(f"resposta {i}", "a", private=i < 5)
    assert await maybe_summarize(orch, conversation)
    prompt = online.calls[0].messages[-1].content
    assert "segredo 0" not in prompt and "segredo 5" in prompt
