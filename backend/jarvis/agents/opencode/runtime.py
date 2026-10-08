"""Shared OpenCode runtime: one sandbox, one private server, many free models.

- ``verify()`` inspects OpenCode through the private server (no model call):
  the only provider must be OpenCode Zen with the anonymous ``public`` key, the
  ``jarvis`` agent must be locked, and it returns the models that are free.
- ``run()`` sends one message with the official client
  ``opencode run --server … --agent jarvis --format json`` and parses its events.

Why ``opencode run`` and not the HTTP prompt route: OpenCode Zen refuses free-tier
requests that do not come from the OpenCode client ("FreeTierError: can only be
used from within OpenCode"). JARVIS respects that and uses the real client.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from jarvis.agents.errors import error_from_message
from jarvis.agents.opencode.sandbox import AGENT_NAME, OpenCodeSandbox, agent_is_locked
from jarvis.agents.opencode.server import OpenCodeServer
from jarvis.core.errors import (
    CostViolationError,
    InvalidResponseError,
    NotConfiguredError,
    ProviderUnavailableError,
)

PROVIDER_ID = "opencode"
PUBLIC_KEY = "public"  # OpenCode Zen's anonymous key: no account, nothing to bill
FREE_MODEL = re.compile(r"(-free|^big-pickle)$")
VERIFY_TTL = 300.0


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


@dataclass
class RunResult:
    text: str
    cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    blocked_tool_calls: list[str] = field(default_factory=list)


def parse_run_output(stdout: str) -> RunResult:
    """Parse ``opencode run --format json`` (one JSON event per line)."""
    texts: dict[str, list[str]] = {}
    order: list[str] = []
    costs: list[float] = []
    tokens_in = tokens_out = 0
    saw_tokens = False
    blocked: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind, part = event.get("type"), event.get("part") or {}
        if kind == "error":
            error = event.get("error") or {}
            raise error_from_message(str(error.get("message") or error))
        if kind == "text":
            message_id = part.get("messageID", "")
            if message_id not in texts:
                texts[message_id] = []
                order.append(message_id)
            texts[message_id].append(part.get("text", ""))
        elif kind == "tool_use":
            state = part.get("state") or {}
            if state.get("status") != "error":
                # A tool ran: the sandbox failed. Never trust this answer.
                raise CostViolationError(f"OpenCode executed tool {part.get('tool')!r}")
            blocked.append(str(part.get("tool")))
        elif kind == "step_finish":
            if part.get("cost") is not None:
                costs.append(float(part["cost"]))
            tokens = part.get("tokens") or {}
            if tokens:
                saw_tokens = True
                tokens_in += int(tokens.get("input") or 0)
                tokens_out += int(tokens.get("output") or 0)
    if not order:
        raise InvalidResponseError("OpenCode finished without an answer")
    text = "".join(texts[order[-1]]).strip()
    if not text:
        raise InvalidResponseError("OpenCode returned an empty answer")
    return RunResult(
        text=text,
        cost=sum(costs) if costs else None,
        input_tokens=tokens_in if saw_tokens else None,
        output_tokens=tokens_out if saw_tokens else None,
        blocked_tool_calls=blocked,
    )


class OpenCodeRuntime:
    def __init__(self, binary: str = "opencode", *, sandbox: OpenCodeSandbox | None = None):
        self.binary = binary
        self._sandbox = sandbox
        self._server: OpenCodeServer | None = None
        self._free_models: set[str] = set()
        self._verified_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def sandbox(self) -> OpenCodeSandbox:
        if self._sandbox is None:
            self._sandbox = OpenCodeSandbox()
        return self._sandbox

    @property
    def server(self) -> OpenCodeServer:
        if self._server is None:
            self._server = OpenCodeServer(self.sandbox, self.binary)
        return self._server

    @property
    def version(self) -> str | None:
        return self._server.version if self._server else None

    async def verify(self, *, force: bool = False) -> set[str]:
        """Check the sandbox and return the ids of the free models (no model call)."""
        async with self._lock:
            fresh = self._verified_at and time.monotonic() - self._verified_at < VERIFY_TTL
            if fresh and not force and self.server.running:
                return self._free_models
            await self.server.ensure_started()

            providers = await self.server.providers()
            ids = sorted(p.get("id") for p in providers)
            if ids != [PROVIDER_ID]:
                raise CostViolationError(f"unexpected OpenCode providers in the sandbox: {ids}")
            api_key = (providers[0].get("settings") or {}).get("apiKey")
            if api_key != PUBLIC_KEY:
                raise CostViolationError("OpenCode Zen is not using the anonymous public key")

            agent = await self.server.agent(AGENT_NAME)
            if not agent_is_locked(agent.get("permissions") or []):
                raise NotConfiguredError("the jarvis agent is not locked (tools could run)")

            self._free_models = {m["id"] for m in await self.server.models() if is_free_model(m)}
            self._verified_at = time.monotonic()
            return self._free_models

    async def run(self, model_id: str, prompt: str, time_limit: float) -> RunResult:
        free = await self.verify()
        if model_id not in free:
            raise ProviderUnavailableError(f"model {model_id} is not offered for free right now")

        server = self.server
        proc = await asyncio.create_subprocess_exec(
            server.executable(),
            "run",
            "--server",
            server.url,
            "--agent",
            AGENT_NAME,
            "--model",
            f"{PROVIDER_ID}/{model_id}",
            "--format",
            "json",
            prompt,
            cwd=self.sandbox.workdir,
            env=self.sandbox.env(OPENCODE_SERVER_PASSWORD=server.password or ""),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), time_limit)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ProviderUnavailableError(f"no answer after {time_limit:.0f}s") from exc

        output = stdout.decode(errors="replace")
        try:
            return parse_run_output(output)
        except InvalidResponseError:
            if proc.returncode:
                detail = stderr.decode(errors="replace").strip()[-300:]
                raise ProviderUnavailableError(
                    f"opencode run exited with {proc.returncode}: {detail}"
                ) from None
            raise

    async def close(self) -> None:
        if self._server is not None:
            await self._server.close()
        if self._sandbox is not None:
            self._sandbox.cleanup()
            self._sandbox = None
            self._server = None
        self._verified_at = None
