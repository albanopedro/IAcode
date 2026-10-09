"""CLI commands, run against a temporary config and data folder (no network, no real data)."""

import builtins

import pytest

from jarvis import cli
from jarvis.core.agent_manager import AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.orchestrator import Orchestrator
from jarvis.memory.assistant import Assistant
from jarvis.memory.store import MemoryStore
from jarvis.tools.builtin.basic import CalculatorTool
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.toolkit import ToolKit
from tests.fakes import FakeAgent, FakeClock

CONFIG = """
[opencode]
enabled = false

[local]
enabled = false

[[openai_compat]]
id = "groq:test"
name = "Groq · teste"
provider = "groq"
base_url = "https://api.groq.com/openai/v1"
model = "openai/gpt-oss-120b"
api_key_env = "JARVIS_TEST_MISSING_KEY"
daily_limit = 10
"""


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    config = tmp_path / "agents.toml"
    config.write_text(CONFIG)
    monkeypatch.setenv("JARVIS_CONFIG", str(config))
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("JARVIS_TEST_MISSING_KEY", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: _settings())
    return tmp_path


def _settings():
    from jarvis.config import load_settings

    return load_settings(env_path=__import__("pathlib").Path("/nonexistent/.env"))


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_status_text_and_json(capsys):
    code, out = run(capsys, "status")
    assert code == 0 and "COST_MODE=FREE_ONLY" in out
    assert "groq:test" in out and "unconfigured" in out and "JARVIS_TEST_MISSING_KEY" in out
    code, out = run(capsys, "status", "--json")
    assert code == 0 and '"cost_mode": "FREE_ONLY"' in out and '"health": "unconfigured"' in out


def test_memory_commands(capsys, monkeypatch):
    assert run(capsys, "memory", "list")[1].strip() == "(nenhuma lembrança guardada)"
    code, out = run(capsys, "memory", "add", "eu", "uso", "macOS")
    assert code == 0 and "guardado: eu uso macOS" in out
    assert "1  eu uso macOS" in run(capsys, "memory", "list")[1]
    assert run(capsys, "memory", "forget", "99")[0] == 1
    assert run(capsys, "memory", "forget", "1")[0] == 0
    monkeypatch.setattr(builtins, "input", lambda prompt: "não")
    assert run(capsys, "memory", "clear")[0] == 1  # anything but "SIM" cancels


def test_history_commands(capsys, monkeypatch, sandbox):
    store = MemoryStore(sandbox / "data" / "memory.db")
    conversation = store.new_conversation()
    conversation.add_user("explique Docker")
    conversation.add_assistant("Docker é uma plataforma.", "a")
    store.close()
    code, out = run(capsys, "history", "list")
    assert code == 0 and "explique Docker" in out
    code, out = run(capsys, "history", "show", conversation.id[:8])
    assert "Você: explique Docker" in out and "JARVIS: Docker é uma plataforma." in out
    assert run(capsys, "history", "show", "zzz")[0] == 1
    monkeypatch.setattr(builtins, "input", lambda prompt: "SIM")
    code, out = run(capsys, "history", "clear")
    assert code == 0 and "1 conversas apagadas" in out


def test_tools_and_local_commands(capsys):
    code, out = run(capsys, "tools", "list")
    assert code == 0 and "run_python" in out and "confirm" in out and "indisponível" in out
    assert "(nenhuma ferramenta usada ainda)" in run(capsys, "tools", "log")[1]
    code, out = run(capsys, "local", "status")
    assert code == 0 and "baixado: não" in out


def test_unblock_command(capsys):
    assert run(capsys, "unblock", "nao-existe")[0] == 2
    assert run(capsys, "unblock", "groq:test")[0] == 1  # nothing to lift


# -- ask / chat with fake agents ----------------------------------------------------------


@pytest.fixture
def fake_build(monkeypatch):
    agent = FakeAgent(
        "a", ['```tool\n{"name": "calculator", "args": {"expression": "6*7"}}\n```', "São 42."]
    )
    manager = AgentManager([agent], CostGuard(), clock=FakeClock())
    kit = ToolKit(ToolExecutor([CalculatorTool()]))
    assistant = Assistant(Orchestrator(manager), MemoryStore(), summarize=False, tools=kit)
    monkeypatch.setattr(cli, "build", lambda: (manager, assistant, 60.0))
    return agent


def test_ask_prints_answer_agent_and_tools(capsys, fake_build):
    code, out = run(capsys, "ask", "quanto", "é", "6*7?")
    assert code == 0
    assert "JARVIS: São 42." in out and "✓ a" in out and "🔧 ✓ Calculadora (allowed)" in out


def test_chat_session(capsys, monkeypatch, fake_build):
    lines = iter(["/status", "quanto é 6*7?", "/limpar", "/sair"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))
    code, out = run(capsys, "chat")
    assert code == 0
    assert "JARVIS pronto" in out and "COST_MODE=FREE_ONLY" in out and "São 42." in out
    assert "nova conversa" in out


async def test_cli_confirmation_answers(monkeypatch, capsys):
    from jarvis.tools.base import ToolCall

    tool = CalculatorTool()
    for typed, expected in (("s", True), ("SIM", True), ("", False), ("n", False)):
        monkeypatch.setattr(builtins, "input", lambda prompt, typed=typed: typed)
        assert (
            await cli.cli_confirm(ToolCall("calculator", {"expression": "1"}), tool, "teste")
            is expected
        )
    assert "O JARVIS quer usar" in capsys.readouterr().out


def test_doctor_reports_without_secrets(capsys, monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_MISSING_KEY", "gsk_super_secret_value")
    code, out = run(capsys, "doctor")
    assert code == 0
    assert "COST_MODE=FREE_ONLY" in out and "1 de 1 chaves" in out
    assert "JARVIS_TEST_MISSING_KEY" in out  # the variable name, never the value
    assert "gsk_super_secret_value" not in out
