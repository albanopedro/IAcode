import json
import os
import stat

import pytest

from jarvis.agents.opencode.adapter import OpenCodeAgent, render_prompt
from jarvis.agents.opencode.runtime import OpenCodeRuntime, is_free_model, parse_run_output
from jarvis.agents.opencode.sandbox import ASK_ALL, OpenCodeSandbox, agent_is_locked
from jarvis.core.errors import (
    CostViolationError,
    InvalidResponseError,
    NotConfiguredError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.core.types import AIRequest, Health, Message

FREE_COST = [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}]
DEFAULT_RULES = [{"action": "*", "resource": "*", "effect": "allow"}]


def model(model_id="space-bunny-free", cost=None):
    return {"id": model_id, "providerID": "opencode", "cost": FREE_COST if cost is None else cost}


def event(kind, **part):
    return json.dumps({"type": kind, "part": part})


def text(message_id, value):
    return event("text", messageID=message_id, text=value)


# -- free model rules ----------------------------------------------------------


def test_free_model_needs_zero_price_and_free_name():
    assert is_free_model(model())
    assert is_free_model(model("big-pickle"))
    assert not is_free_model(model("claude-opus"))  # not on the free allowlist
    assert not is_free_model(model(cost=[{"input": 0.5, "output": 2}]))
    assert not is_free_model(model(cost=[{"input": 0, "output": 0, "cache": {"read": 0.1}}]))
    assert not is_free_model(model(cost=[]))  # unknown price is not free
    assert not is_free_model({**model(), "providerID": "anthropic"})


# -- prompt ----------------------------------------------------------------------


def test_render_prompt_includes_history_and_current_message():
    prompt = render_prompt(
        [
            Message(role="system", content="Seja breve."),
            Message(role="user", content="explique Docker"),
            Message(role="assistant", content="Docker é..."),
            Message(role="user", content="e no Mac?"),
        ]
    )
    assert "Seja breve." in prompt
    assert "Usuário: explique Docker" in prompt
    assert "JARVIS: Docker é..." in prompt
    assert prompt.index("# Mensagem atual do usuário") < prompt.index("e no Mac?")
    assert "ferramentas nativas" in prompt and "```tool```" in prompt


def test_render_prompt_requires_a_user_message_last():
    with pytest.raises(ValueError):
        render_prompt([Message(role="assistant", content="oi")])


# -- run output ----------------------------------------------------------------


def test_parse_keeps_only_the_final_message_text_and_sums_cost():
    out = "\n".join(
        [
            "! permission requested: read (x); auto-rejecting",  # non-JSON noise
            event("step_start"),
            text("m1", "Vou tentar ler..."),
            event("tool_use", tool="read", state={"status": "error", "error": "rejected"}),
            event("step_finish", cost=0, tokens={"input": 100, "output": 5}),
            text("m2", "Olá, "),
            text("m2", "sou o JARVIS."),
            event("step_finish", cost=0, tokens={"input": 50, "output": 4}),
        ]
    )
    result = parse_run_output(out)
    assert result.text == "Olá, sou o JARVIS."
    assert result.cost == 0
    assert (result.input_tokens, result.output_tokens) == (150, 9)
    assert result.blocked_tool_calls == ["read"]


def test_parse_without_step_finish_reports_unknown_cost():
    result = parse_run_output(text("m1", "ok"))
    assert result.cost is None and result.input_tokens is None


def test_parse_reports_a_paid_step():
    out = "\n".join([text("m1", "ok"), event("step_finish", cost=0.003)])
    assert parse_run_output(out).cost == pytest.approx(0.003)


def test_a_tool_that_actually_ran_is_a_violation():
    out = "\n".join(
        [event("tool_use", tool="shell", state={"status": "completed"}), text("m1", "feito")]
    )
    with pytest.raises(CostViolationError, match="executed tool"):
        parse_run_output(out)


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("Rate limit exceeded", RateLimitError),
        ("OpenCode's free tier can only be used from within OpenCode", NotConfiguredError),
        ("upstream exploded", ProviderUnavailableError),
    ],
)
def test_error_events_are_classified(message, error):
    out = json.dumps({"type": "error", "error": {"type": "provider", "message": message}})
    with pytest.raises(error):
        parse_run_output(out)


def test_no_text_is_invalid():
    with pytest.raises(InvalidResponseError):
        parse_run_output(event("step_finish", cost=0))


# -- sandbox -------------------------------------------------------------------


def test_sandbox_env_is_an_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENCODE_PERMISSION", '{"*":"allow"}')
    monkeypatch.setenv("PATH", "/usr/bin")
    sandbox = OpenCodeSandbox(str(tmp_path))
    env = sandbox.env(EXTRA="1")
    assert "OPENAI_API_KEY" not in env
    assert "OPENCODE_PERMISSION" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["PWD"] == str(sandbox.workdir)
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "config")
    assert env["EXTRA"] == "1"
    config = json.loads((tmp_path / "config" / "opencode" / "opencode.json").read_text())
    assert config["agents"]["jarvis"]["permissions"] == [ASK_ALL]
    assert list(sandbox.workdir.iterdir()) == []  # nothing to read in the work folder


def test_sandbox_cleanup_only_removes_its_own_temp_root(tmp_path):
    given = OpenCodeSandbox(str(tmp_path))
    given.cleanup()
    assert tmp_path.exists()
    owned = OpenCodeSandbox()
    root = owned.root
    owned.cleanup()
    assert not root.exists()


def test_agent_lock_requires_ask_all_as_the_last_rule():
    assert agent_is_locked([*DEFAULT_RULES, ASK_ALL])
    assert not agent_is_locked([ASK_ALL, *DEFAULT_RULES])  # a later allow wins
    assert not agent_is_locked(DEFAULT_RULES)
    assert not agent_is_locked([])


# -- runtime verify --------------------------------------------------------------


class FakeServer:
    def __init__(self, *, providers=None, permissions=None, models=None):
        public = [{"id": "opencode", "settings": {"apiKey": "public"}}]
        self.providers_list = public if providers is None else providers
        self.permissions = [*DEFAULT_RULES, ASK_ALL] if permissions is None else permissions
        self.models_list = [model()] if models is None else models
        self.running = True
        self.url = "http://127.0.0.1:1"
        self.password = "pw"
        self.version = "2.0.20"
        self.binary = "opencode"
        self.closed = False

    def executable(self):
        return self.binary

    async def ensure_started(self):
        pass

    async def providers(self):
        return self.providers_list

    async def agent(self, name):
        assert name == "jarvis"
        return {"id": name, "permissions": self.permissions}

    async def models(self):
        return self.models_list

    async def close(self):
        self.closed = True


def runtime_with(server, tmp_path):
    runtime = OpenCodeRuntime(sandbox=OpenCodeSandbox(str(tmp_path)))
    runtime._server = server
    return runtime


async def test_verify_returns_free_models(tmp_path):
    server = FakeServer(models=[model(), model("pro", cost=[{"input": 1, "output": 1}])])
    assert await runtime_with(server, tmp_path).verify() == {"space-bunny-free"}


@pytest.mark.parametrize(
    "providers",
    [
        [{"id": "opencode", "settings": {"apiKey": "sk-paid"}}],
        [{"id": "opencode", "settings": {"apiKey": "public"}}, {"id": "anthropic"}],
    ],
)
async def test_verify_blocks_paid_credentials_or_extra_providers(tmp_path, providers):
    with pytest.raises(CostViolationError):
        await runtime_with(FakeServer(providers=providers), tmp_path).verify()


async def test_verify_refuses_an_unlocked_agent(tmp_path):
    with pytest.raises(NotConfiguredError, match="not locked"):
        await runtime_with(FakeServer(permissions=DEFAULT_RULES), tmp_path).verify()


# -- runtime run (with a fake `opencode` executable) ------------------------------


def fake_opencode(tmp_path, stdout_lines, exit_code=0, sleep=0):
    """A shell script that records its argv and env, then prints canned events."""
    script = tmp_path / "fake-opencode"
    record = tmp_path / "record.txt"
    payload = "\n".join(stdout_lines).replace("'", "'\\''")
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{record}"\n'
        f'echo "PWD=$PWD" >> "{record}"\n'
        f'echo "KEY=${{OPENAI_API_KEY:-none}}" >> "{record}"\n'
        f'echo "SERVERPW=$OPENCODE_SERVER_PASSWORD" >> "{record}"\n'
        f"sleep {sleep}\n"
        f"printf '%s\\n' '{payload}'\n"
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), record


async def test_run_uses_the_official_client_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    binary, record = fake_opencode(tmp_path, [text("m1", "Olá!"), event("step_finish", cost=0)])
    server = FakeServer()
    server.binary = binary
    runtime = runtime_with(server, tmp_path / "home")
    result = await runtime.run("space-bunny-free", "oi", time_limit=10)
    assert result.text == "Olá!"

    recorded = record.read_text().splitlines()
    args = recorded[: recorded.index("oi") + 1]
    assert args[:2] == ["run", "--server"]
    assert args[args.index("--agent") + 1] == "jarvis"
    assert args[args.index("--model") + 1] == "opencode/space-bunny-free"
    assert "--auto" not in args  # would auto-approve tools
    assert f"PWD={runtime.sandbox.workdir}" in recorded
    assert "KEY=none" in recorded
    assert "SERVERPW=pw" in recorded


async def test_run_refuses_models_that_are_not_free_now(tmp_path):
    runtime = runtime_with(FakeServer(models=[model("other-free")]), tmp_path)
    with pytest.raises(ProviderUnavailableError, match="not offered"):
        await runtime.run("space-bunny-free", "oi", time_limit=10)


async def test_run_timeout_kills_the_client(tmp_path):
    binary, _ = fake_opencode(tmp_path, [text("m1", "tarde demais")], sleep=5)
    server = FakeServer()
    server.binary = binary
    with pytest.raises(ProviderUnavailableError, match="no answer"):
        await runtime_with(server, tmp_path / "home").run("space-bunny-free", "oi", time_limit=0.3)


async def test_run_crash_without_output_is_unavailable(tmp_path):
    binary, _ = fake_opencode(tmp_path, [], exit_code=3)
    server = FakeServer()
    server.binary = binary
    with pytest.raises(ProviderUnavailableError, match="exited with 3"):
        await runtime_with(server, tmp_path / "home").run("space-bunny-free", "oi", time_limit=10)


# -- agent -----------------------------------------------------------------------


async def test_agent_generate_and_health(tmp_path):
    binary, _ = fake_opencode(tmp_path, [text("m1", "Oi!"), event("step_finish", cost=0)])
    server = FakeServer()
    server.binary = binary
    runtime = runtime_with(server, tmp_path / "home")
    agent = OpenCodeAgent(runtime, "space-bunny-free")
    request = AIRequest(messages=[Message(role="user", content="oi")])
    response = await agent.generate(request)
    assert (response.text, response.agent_id, response.cost) == (
        "Oi!",
        "opencode:space-bunny-free",
        0,
    )
    assert (await agent.check_health()).health is Health.AVAILABLE
    gone = OpenCodeAgent(runtime, "withdrawn-free")
    assert (await gone.check_health()).health is Health.OFFLINE


async def test_agent_health_blocked_on_paid_credentials(tmp_path):
    server = FakeServer(providers=[{"id": "opencode", "settings": {"apiKey": "sk"}}])
    agent = OpenCodeAgent(runtime_with(server, tmp_path), "space-bunny-free")
    assert (await agent.check_health()).health is Health.BLOCKED


def test_fake_binary_helper_is_executable(tmp_path):
    binary, _ = fake_opencode(tmp_path, ["x"])
    assert os.access(binary, os.X_OK)
