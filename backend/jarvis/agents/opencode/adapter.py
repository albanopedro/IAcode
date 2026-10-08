"""OpenCode adapter: each free OpenCode Zen model becomes one JARVIS agent.

All agents share one OpenCodeRuntime (sandbox + private server). See
``runtime.py`` and ``sandbox.py`` for how cost and tool safety are enforced.
"""

from __future__ import annotations

import time

from jarvis.agents.opencode.runtime import PROVIDER_ID, OpenCodeRuntime
from jarvis.core.errors import CostViolationError, NotConfiguredError, ProviderError
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
    TaskType,
)


def render_prompt(messages: list[Message]) -> str:
    """Flatten the conversation into one prompt (``opencode run`` takes one message)."""
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
        "Responda diretamente com texto: não use ferramentas, não leia arquivos e "
        "não execute comandos.",
    ]
    return "\n".join(parts)


class OpenCodeAgent(AIProvider):
    def __init__(
        self,
        runtime: OpenCodeRuntime,
        model_id: str,
        *,
        name: str | None = None,
        priority: int = 50,
        capabilities: frozenset[Capability] = frozenset({Capability.CHAT}),
        privacy: Privacy = Privacy.UNKNOWN,
        quality: dict[TaskType, float] | None = None,
        rpm_limit: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.timeout = timeout
        self.info = AgentInfo(
            id=f"{PROVIDER_ID}:{model_id}",
            name=name or f"OpenCode · {model_id}",
            provider=PROVIDER_ID,
            model=model_id,
            cost_class=CostClass.FREE,
            capabilities=capabilities,
            priority=priority,
            privacy=privacy,
            quality=quality or {},
            rpm_limit=rpm_limit,
        )

    @property
    def context_window(self) -> int | None:
        return self.runtime.context_window(self.model_id) or self.info.context_window

    async def check_health(self) -> HealthReport:
        try:
            free = await self.runtime.verify()
        except CostViolationError as exc:
            return HealthReport(Health.BLOCKED, str(exc))
        except NotConfiguredError as exc:
            return HealthReport(Health.UNCONFIGURED, str(exc))
        except ProviderError as exc:
            return HealthReport(Health.OFFLINE, str(exc))
        if self.model_id not in free:
            return HealthReport(Health.OFFLINE, f"{self.model_id} is not offered for free now")
        return HealthReport(Health.AVAILABLE, f"OpenCode {self.runtime.version}")

    async def generate(self, request: AIRequest) -> AIResponse:
        started = time.perf_counter()
        result = await self.runtime.run(
            self.model_id, render_prompt(request.messages), self.timeout
        )
        return AIResponse(
            text=result.text,
            agent_id=self.id,
            model=self.model_id,
            cost=result.cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def close(self) -> None:
        await self.runtime.close()
