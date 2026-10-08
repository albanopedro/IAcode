import pytest

from jarvis.agents.opencode.adapter import DENY_ALL, OpenCodeAgent, is_free_model, render_prompt
from jarvis.core.errors import (
    CostViolationError,
    NotConfiguredError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.core.types import AIRequest, Health, Message

FREE_COST = [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}]


def model(model_id="space-bunny-free", cost=None, status="active"):
    return {
        "id": model_id,
        "providerID": "opencode",
        "cost": FREE_COST if cost is None else cost,
        "status": status,
        "enabled": True,
    }


class FakeServer:
    """Mimics the OpenCode v2 HTTP API surface the adapter uses."""

    def __init__(self, *, models=None, api_key="public", reply="Olá!", outcome="succeeded"):
        self.models_list = [model()] if models is None else models
        self.api_key = api_key
        self.reply = reply
        self.outcome = outcome
        self.assistant_error = None
        self.session_permissions = DENY_ALL
        self.session_cost = 0
        self.never_idle = False
        self.requests: list[tuple[str, str, object]] = []
        self.deleted: list[str] = []
        self.workdir = "/tmp/jarvis-opencode-test"
        self.version = "2.0.20"

    async def ensure_started(self):
        pass

    async def models(self):
        return self.models_list

    async def provider(self, provider_id):
        return {"id": provider_id, "settings": {"apiKey": self.api_key}}

    async def request(self, method, path, json=None):
        self.requests.append((method, path, json))
        if method == "POST" and path == "/api/session":
            return {"data": {"id": "ses_1", "permissions": self.session_permissions}}
        if method == "POST" and path.endswith("/prompt"):
            return {"data": {"id": "msg_u"}}
        if method == "POST" and path.endswith("/interrupt"):
            return None
        if method == "GET" and path == "/api/session/ses_1":
            if self.never_idle:
                return {"data": {"time": {}}}
            return {
                "data": {
                    "time": {"idle": 1},
                    "outcome": self.outcome,
                    "cost": self.session_cost,
                    "tokens": {"input": 10, "output": 3},
                }
            }
        if method == "GET" and path.endswith("/message"):
            assistant = {
                "type": "assistant",
                "time": {"created": 2},
                "content": [
                    {"type": "reasoning", "text": "pensando..."},
                    {"type": "text", "text": self.reply},
                ],
            }
            if self.assistant_error:
                assistant["error"] = {"message": self.assistant_error}
            return {"data": [{"type": "user", "text": "oi"}, assistant]}
        if method == "DELETE":
            self.deleted.append(path)
            return None
        raise AssertionError(f"unexpected request {method} {path}")

    async def close(self):
        pass


def request(*messages):
    return AIRequest(messages=[Message(role="system", content="Você é o JARVIS."), *messages])


def user(text):
    return Message(role="user", content=text)


# -- pure helpers -------------------------------------------------------------


def test_free_model_needs_zero_price_and_free_name():
    assert is_free_model(model())
    assert is_free_model(model("big-pickle"))
    assert not is_free_model(model("claude-opus"))  # not on the free allowlist
    assert not is_free_model(model(cost=[{"input": 0.5, "output": 2}]))
    assert not is_free_model(model(cost=[{"input": 0, "output": 0, "cache": {"read": 0.1}}]))
    assert not is_free_model(model(cost=[]))  # unknown price is not free
    assert not is_free_model({**model(), "providerID": "anthropic"})


def test_render_prompt_includes_history_and_current_message():
    prompt = render_prompt(
        [
            Message(role="system", content="Seja breve."),
            user("explique Docker"),
            Message(role="assistant", content="Docker é..."),
            user("e no Mac?"),
        ]
    )
    assert "Seja breve." in prompt
    assert "Usuário: explique Docker" in prompt
    assert "JARVIS: Docker é..." in prompt
    assert prompt.index("# Mensagem atual do usuário") < prompt.index("e no Mac?")


def test_render_prompt_requires_a_user_message_last():
    with pytest.raises(ValueError):
        render_prompt([Message(role="assistant", content="oi")])


# -- generation ---------------------------------------------------------------


async def test_generate_uses_a_deny_all_session_and_deletes_it():
    server = FakeServer(reply="Olá, eu sou o JARVIS.")
    agent = OpenCodeAgent(server, "space-bunny-free")
    response = await agent.generate(request(user("oi")))

    assert response.text == "Olá, eu sou o JARVIS."  # reasoning is not included
    assert response.agent_id == "opencode:space-bunny-free"
    assert response.cost == 0
    assert (response.input_tokens, response.output_tokens) == (10, 3)
    create = next(r for r in server.requests if r[1] == "/api/session")
    assert create[2]["permissions"] == DENY_ALL
    assert create[2]["model"] == {"providerID": "opencode", "id": "space-bunny-free"}
    assert create[2]["location"] == {"directory": server.workdir}
    assert server.deleted == ["/api/session/ses_1"]


async def test_refuses_when_permissions_are_not_applied():
    server = FakeServer()
    server.session_permissions = []
    with pytest.raises(NotConfiguredError, match="deny-all"):
        await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))
    assert not any(path.endswith("/prompt") for _, path, _ in server.requests)
    assert server.deleted == ["/api/session/ses_1"]


async def test_refuses_when_a_paid_credential_is_configured():
    server = FakeServer(api_key="sk-paid")
    with pytest.raises(CostViolationError, match="non-public credential"):
        await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))
    assert server.requests == []


async def test_refuses_a_model_that_is_no_longer_free():
    server = FakeServer(models=[model(cost=[{"input": 1, "output": 1}])])
    with pytest.raises(CostViolationError):
        await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))


async def test_model_withdrawn_from_the_catalog_is_unavailable():
    server = FakeServer(models=[model("other-free")])
    with pytest.raises(ProviderUnavailableError, match="not offered"):
        await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))


async def test_rate_limit_error_from_the_provider_is_classified():
    server = FakeServer(outcome="failed")
    server.assistant_error = "Rate limit exceeded, try again later"
    with pytest.raises(RateLimitError):
        await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))
    assert server.deleted == ["/api/session/ses_1"]


async def test_reports_session_cost_for_the_cost_guard():
    server = FakeServer()
    server.session_cost = 0.01
    response = await OpenCodeAgent(server, "space-bunny-free").generate(request(user("oi")))
    assert response.cost == 0.01


async def test_timeout_interrupts_the_session():
    server = FakeServer()
    server.never_idle = True
    agent = OpenCodeAgent(server, "space-bunny-free", timeout=0.05, poll_interval=0.01)
    with pytest.raises(ProviderUnavailableError, match="no answer"):
        await agent.generate(request(user("oi")))
    assert any(path.endswith("/interrupt") for _, path, _ in server.requests)
    assert server.deleted == ["/api/session/ses_1"]


# -- health -------------------------------------------------------------------


async def test_health_reports():
    assert (await OpenCodeAgent(FakeServer(), "space-bunny-free").check_health()).health is (
        Health.AVAILABLE
    )
    paid = OpenCodeAgent(FakeServer(api_key="sk-x"), "space-bunny-free")
    assert (await paid.check_health()).health is Health.BLOCKED
    gone = OpenCodeAgent(FakeServer(models=[]), "space-bunny-free")
    assert (await gone.check_health()).health is Health.OFFLINE
