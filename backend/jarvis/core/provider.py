"""The interface every agent adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jarvis.core.types import AgentInfo, AIRequest, AIResponse, Capability, Health


@dataclass(frozen=True)
class HealthReport:
    health: Health
    detail: str = ""


class AIProvider(ABC):
    """An agent JARVIS can talk to.

    Adapters only describe themselves (``info``) and talk to their backend.
    Runtime state — cooldowns, error counts, latency — lives in the AgentManager,
    because it is observed from the outside, call after call.
    """

    info: AgentInfo

    @property
    def id(self) -> str:
        return self.info.id

    def supports(self, capability: Capability) -> bool:
        return capability in self.info.capabilities

    @property
    def context_window(self) -> int | None:
        """Tokens the model accepts. Adapters that discover it at runtime override this."""
        return self.info.context_window

    @abstractmethod
    async def check_health(self) -> HealthReport:
        """Cheap check that never spends quota (no generation)."""

    @abstractmethod
    async def generate(self, request: AIRequest) -> AIResponse:
        """Answer the request or raise a ProviderError subclass."""

    async def close(self) -> None:  # noqa: B027 — optional hook
        """Release resources (processes, HTTP clients)."""
