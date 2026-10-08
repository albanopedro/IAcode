"""OpenCode adapter: each free OpenCode Zen model becomes one JARVIS agent.

All agents share one private OpenCode server. Every request:
1. checks the model is still offered with zero cost and no paid credential is set;
2. creates a throw-away session with ALL permissions denied (no tools at all);
3. sends the conversation as one prompt and waits for the session to go idle;
4. reads the answer and the reported cost, then deletes the session.

The stateless ``/api/experimental/generate`` route is NOT used: OpenCode Zen
refuses free-tier calls from it ("can only be used from within OpenCode").
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time

from jarvis.agents.errors import error_from_message
from jarvis.agents.opencode.server import OpenCodeServer
from jarvis.core.errors import (
    CostViolationError,
    InvalidResponseError,
    JarvisError,
    NotConfiguredError,
    ProviderError,
    ProviderUnavailableError,
)
from jarvis.core.provider import AIProvider, HealthReport
from jarvis.core.types import (
    AgentInfo,
    AIRequest,
    AIResponse,
    Capability,
    CostClass,
    Health,
    Message,
    Privacy,
)

PROVIDER_ID = "opencode"
PUBLIC_KEY = "public"  # the anonymous Zen key: no paid credential attached
FREE_MODEL = re.compile(r"(-free|^big-pickle)$")
DENY_ALL = [{"action": "*", "resource": "*", "effect": "deny"}]


def is_free_model(model: dict) -> bool:
    """Zero price on every cost tier AND a name from the free allowlist."""
    if model.get("providerID") != PROVIDER_ID or not FREE_MODEL.search(model.get("id", "")):
        return False
    costs = model.get("cost") or []
    if not costs:
        return False
    for tier in costs:
        cache = tier.get("cache") or {}
        prices = [
            tier.get("input"),
            tier.get("output"),
            cache.get("read", 0),
            cache.get("write", 0),
        ]
        if any(price is None or price != 0 for price in prices):
            return False
    return True


def render_prompt(messages: list[Message]) -> str:
    """Flatten the conversation into one prompt (OpenCode takes a single text)."""
    system = "\n".join(m.content for m in messages if m.role == "system")
    dialogue = [m for m in messages if m.role != "system"]
    if not dialogue or dialogue[-1].role != "user":
        raise ValueError("the last message must come from the user")
    *history, current = dialogue

    parts = ["# Instruções", system, ""]
    if history:
        parts.append("# Conversa até agora")
        for message in history:
            speaker = "Usuário" if message.role == "user" else "JARVIS"
            parts.append(f"{speaker}: {message.content}")
        parts.append("")
    parts += [
        "# Mensagem atual do usuário",
        current.content,
        "",
        "Responda apenas à mensagem atual, levando em conta a conversa acima. "
        "Responda somente com texto.",
    ]
    return "\n".join(parts)


class OpenCodeAgent(AIProvider):
    def __init__(
        self,
        server: OpenCodeServer,
        model_id: str,
        *,
        name: str | None = None,
        priority: int = 50,
        capabilities: frozenset[Capability] = frozenset({Capability.CHAT}),
        privacy: Privacy = Privacy.UNKNOWN,
        timeout: float = 120.0,
        poll_interval: float = 0.4,
    ) -> None:
        self.server = server
        self.model_id = model_id
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.info = AgentInfo(
            id=f"{PROVIDER_ID}:{model_id}",
            name=name or f"OpenCode · {model_id}",
            provider=PROVIDER_ID,
            model=model_id,
            cost_class=CostClass.FREE,
            capabilities=capabilities,
            priority=priority,
            privacy=privacy,
        )

    # -- safety checks -------------------------------------------------------

    async def _verify_free(self) -> None:
        """Raise unless this model is free AND OpenCode has no paid credential."""
        provider = await self.server.provider(PROVIDER_ID)
        api_key = (provider.get("settings") or {}).get("apiKey")
        if api_key != PUBLIC_KEY:
            raise CostViolationError(
                "OpenCode Zen has a non-public credential configured; "
                "refusing to use it while COST_MODE=FREE_ONLY"
            )
        models = await self.server.models()
        model = next(
            (m for m in models if m.get("providerID") == PROVIDER_ID and m["id"] == self.model_id),
            None,
        )
        if model is None or model.get("status") != "active" or not model.get("enabled", True):
            raise ProviderUnavailableError(f"model {self.model_id} is not offered right now")
        if not is_free_model(model):
            raise CostViolationError(
                f"model {self.model_id} is not free (cost={model.get('cost')})"
            )

    async def check_health(self) -> HealthReport:
        try:
            await self.server.ensure_started()
            await self._verify_free()
        except CostViolationError as exc:
            return HealthReport(Health.BLOCKED, str(exc))
        except NotConfiguredError as exc:
            return HealthReport(Health.UNCONFIGURED, str(exc))
        except ProviderError as exc:
            return HealthReport(Health.OFFLINE, str(exc))
        return HealthReport(Health.AVAILABLE, f"OpenCode {self.server.version}")

    # -- generation -----------------------------------------------------------

    async def generate(self, request: AIRequest) -> AIResponse:
        started = time.perf_counter()
        await self.server.ensure_started()
        await self._verify_free()

        created = await self.server.request(
            "POST",
            "/api/session",
            {
                "title": "jarvis",
                "model": {"providerID": PROVIDER_ID, "id": self.model_id},
                "location": {"directory": self.server.workdir},
                "permissions": DENY_ALL,
            },
        )
        session = created["data"]
        session_id = session["id"]
        try:
            if session.get("permissions") != DENY_ALL:
                raise NotConfiguredError(
                    "OpenCode did not accept the deny-all permission rules; refusing to run"
                )
            await self.server.request(
                "POST",
                f"/api/session/{session_id}/prompt",
                {"text": render_prompt(request.messages)},
            )
            final = await self._wait_idle(session_id)
            text = await self._read_answer(session_id, final)
        finally:
            await self._delete_session(session_id)

        tokens = final.get("tokens") or {}
        return AIResponse(
            text=text,
            agent_id=self.id,
            model=self.model_id,
            cost=float(final.get("cost") or 0),
            input_tokens=tokens.get("input"),
            output_tokens=tokens.get("output"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def _wait_idle(self, session_id: str) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while True:
            session = (await self.server.request("GET", f"/api/session/{session_id}"))["data"]
            if (session.get("time") or {}).get("idle") or session.get("outcome"):
                return session
            if loop.time() > deadline:
                with contextlib.suppress(JarvisError):
                    await self.server.request("POST", f"/api/session/{session_id}/interrupt", {})
                raise ProviderUnavailableError(f"no answer after {self.timeout:.0f}s")
            await asyncio.sleep(self.poll_interval)

    async def _read_answer(self, session_id: str, session: dict) -> str:
        messages = (await self.server.request("GET", f"/api/session/{session_id}/message"))["data"]
        assistant = [m for m in messages if m.get("type") == "assistant"]
        if session.get("outcome") not in (None, "succeeded"):
            raise error_from_message(self._error_text(session, assistant))
        if not assistant:
            raise InvalidResponseError("OpenCode finished without an assistant message")
        latest = max(assistant, key=lambda m: (m.get("time") or {}).get("created", 0))
        if latest.get("error"):
            raise error_from_message(self._error_text(session, [latest]))
        text = "".join(
            part.get("text", "")
            for part in latest.get("content") or []
            if part.get("type") == "text"
        ).strip()
        if not text:
            raise InvalidResponseError("OpenCode returned an empty answer")
        return text

    @staticmethod
    def _error_text(session: dict, assistant: list[dict]) -> str:
        for message in assistant:
            error = message.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
        return f"OpenCode session ended with outcome={session.get('outcome')}"

    async def _delete_session(self, session_id: str) -> None:
        # Best effort: the temp workspace is removed on shutdown anyway.
        with contextlib.suppress(JarvisError):
            await self.server.request("DELETE", f"/api/session/{session_id}")

    async def close(self) -> None:
        await self.server.close()
