import httpx
import pytest
import respx

from jarvis.agents.openai_compat.adapter import OpenAICompatAgent
from jarvis.core.errors import (
    CostViolationError,
    NotConfiguredError,
    PaymentRequiredError,
    ProviderUnavailableError,
    RateLimitError,
)
from jarvis.core.types import AgentInfo, AIRequest, CostClass, Health, Message

BASE = "https://api.example.test/v1"
KEY_ENV = "JARVIS_TEST_PROVIDER_KEY"


def make(model="some/model:free", pattern=":free$"):
    info = AgentInfo(
        id="ex:1",
        name="Example",
        provider="example",
        model=model,
        cost_class=CostClass.FREE_WITH_LIMITS,
    )
    return OpenAICompatAgent(info, base_url=BASE, api_key_env=KEY_ENV, free_model_pattern=pattern)


REQUEST = AIRequest(messages=[Message(role="user", content="oi")])


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "test-key-123")


@respx.mock
async def test_success_parses_text_usage_and_sends_bearer_key(key):
    route = respx.post(f"{BASE}/chat/completions").respond(
        200,
        json={
            "choices": [{"message": {"content": " Olá! "}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0},
        },
    )
    response = await make().generate(REQUEST)
    assert response.text == "Olá!"
    assert response.cost == 0
    assert (response.input_tokens, response.output_tokens) == (5, 2)
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-key-123"


@respx.mock
async def test_429_becomes_rate_limit_with_retry_after(key):
    respx.post(f"{BASE}/chat/completions").respond(
        429, headers={"retry-after": "42"}, json={"error": {"message": "slow down"}}
    )
    with pytest.raises(RateLimitError) as info:
        await make().generate(REQUEST)
    assert info.value.retry_after == 42


@respx.mock
async def test_402_becomes_payment_required(key):
    respx.post(f"{BASE}/chat/completions").respond(402, json={"error": {"message": "no credits"}})
    with pytest.raises(PaymentRequiredError):
        await make().generate(REQUEST)


@respx.mock
async def test_5xx_and_network_errors_are_unavailable(key):
    respx.post(f"{BASE}/chat/completions").respond(503, text="down")
    with pytest.raises(ProviderUnavailableError):
        await make().generate(REQUEST)
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(ProviderUnavailableError):
        await make().generate(REQUEST)


async def test_missing_key_is_not_configured(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    agent = make()
    assert (await agent.check_health()).health is Health.UNCONFIGURED
    with pytest.raises(NotConfiguredError, match=KEY_ENV):
        await agent.generate(REQUEST)


@respx.mock
async def test_model_outside_the_free_pattern_is_blocked_before_any_call(key):
    route = respx.post(f"{BASE}/chat/completions").respond(200, json={})
    agent = make(model="some/paid-model")
    assert (await agent.check_health()).health is Health.BLOCKED
    with pytest.raises(CostViolationError):
        await agent.generate(REQUEST)
    assert not route.called


@respx.mock
async def test_health_checks_the_model_list(key):
    respx.get(f"{BASE}/models").respond(200, json={"data": [{"id": "some/model:free"}]})
    assert (await make().check_health()).health is Health.AVAILABLE
    respx.get(f"{BASE}/models").respond(200, json={"data": [{"id": "other"}]})
    assert (await make().check_health()).health is Health.OFFLINE
    respx.get(f"{BASE}/models").respond(401, json={})
    assert (await make().check_health()).health is Health.UNCONFIGURED


@respx.mock
async def test_quota_headers_are_reported(key):
    respx.post(f"{BASE}/chat/completions").respond(
        200,
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-reset-requests": "1m",
        },
        json={"choices": [{"message": {"content": "ok"}}]},
    )
    response = await make().generate(REQUEST)
    assert (response.rate_limit.remaining, response.rate_limit.reset_seconds) == (999, 60)


@respx.mock
async def test_429_without_retry_after_uses_the_quota_reset(key):
    respx.post(f"{BASE}/chat/completions").respond(
        429,
        headers={"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m"},
        json={"error": {"message": "limit"}},
    )
    with pytest.raises(RateLimitError) as info:
        await make().generate(REQUEST)
    assert info.value.retry_after == 120


def cloudflare_like():
    info = AgentInfo(
        id="cf:1",
        name="CF",
        provider="cloudflare",
        model="@cf/meta/llama-3.1-8b-instruct",
        cost_class=CostClass.FREE_WITH_LIMITS,
    )
    return OpenAICompatAgent(
        info,
        base_url="https://cf.test/accounts/${JARVIS_TEST_ACCOUNT}/ai/v1",
        api_key_env=KEY_ENV,
        health_check="key",
    )


@respx.mock
async def test_base_url_placeholders_come_from_the_environment(key, monkeypatch):
    monkeypatch.delenv("JARVIS_TEST_ACCOUNT", raising=False)
    agent = cloudflare_like()
    assert (await agent.check_health()).health is Health.UNCONFIGURED
    with pytest.raises(NotConfiguredError, match="JARVIS_TEST_ACCOUNT"):
        await agent.generate(REQUEST)

    monkeypatch.setenv("JARVIS_TEST_ACCOUNT", "acc123")
    route = respx.post("https://cf.test/accounts/acc123/ai/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "oi"}}]}
    )
    assert (await agent.check_health()).health is Health.AVAILABLE  # key-only: no request
    assert (await agent.generate(REQUEST)).text == "oi"
    assert route.called


def test_unknown_health_check_mode_is_rejected():
    with pytest.raises(ValueError):
        OpenAICompatAgent(make().info, base_url=BASE, api_key_env=KEY_ENV, health_check="ping")
