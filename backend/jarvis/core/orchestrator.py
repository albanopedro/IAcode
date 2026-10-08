"""AI Orchestrator: the brain that turns a user message into an answer.

user text → classify → rank free agents → try the best one
          → on failure, record it and try the next → answer (or a clear error)

``private_context`` (e.g. the user's long-term memories) is added to the system
prompt only for agents allowed to see it: by default, local and zero-retention
agents. The other agents still answer, just without that context.
"""

from __future__ import annotations

import time
from enum import StrEnum

from jarvis.core.agent_manager import AgentManager
from jarvis.core.classifier import classify
from jarvis.core.conversation import Conversation
from jarvis.core.errors import AllAgentsFailedError, ProviderError
from jarvis.core.provider import AIProvider
from jarvis.core.router import Router
from jarvis.core.types import (
    AIRequest,
    Attempt,
    Message,
    OrchestratorResult,
    Privacy,
    RankedAgent,
    TaskType,
)

SYSTEM_PROMPT = (
    "Você é o JARVIS, um assistente pessoal inteligente, educado e direto. "
    "Responda no idioma do usuário (português do Brasil por padrão), de forma clara "
    "e concisa, pois suas respostas também poderão ser faladas em voz alta. "
    "Você não tem acesso a arquivos, comandos ou internet nesta conversa."
)
PRIVATE_PRIVACY = frozenset({Privacy.LOCAL, Privacy.ZERO_RETENTION})


class ShareWith(StrEnum):
    PRIVATE = "private"  # only local / zero-retention agents receive private context
    ALL = "all"


class Orchestrator:
    def __init__(
        self,
        manager: AgentManager,
        router: Router | None = None,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_attempts: int = 5,
        share_private_with: ShareWith = ShareWith.PRIVATE,
    ) -> None:
        self.manager = manager
        self.router = router or Router(manager)
        self.system_prompt = system_prompt
        self.max_attempts = max_attempts
        self.share_private_with = share_private_with

    def may_see_private(self, provider: AIProvider) -> bool:
        return self.share_private_with is ShareWith.ALL or provider.info.privacy in PRIVATE_PRIVACY

    def _system(self, provider: AIProvider, style: str | None, private: str | None) -> Message:
        parts = [self.system_prompt]
        if private and self.may_see_private(provider):
            parts.append(private)
        if style:
            parts.append(style)
        return Message(role="system", content="\n\n".join(parts))

    async def complete(
        self,
        messages: list[Message],
        task: TaskType = TaskType.CHAT,
        *,
        style: str | None = None,
        private_context: str | None = None,
    ) -> OrchestratorResult:
        """Stateless: rank the free agents and try them in order until one answers."""
        await self.manager.refresh_health()
        # Rank with the largest possible prompt, so the context-window check is safe.
        extra = "\n\n".join(p for p in (self.system_prompt, private_context, style) if p)
        largest = AIRequest(messages=[Message(role="system", content=extra), *messages], task=task)
        ranked = self.router.rank(self.manager.candidates(largest.required), task, largest)
        ranking = [
            RankedAgent(agent_id=p.id, score=round(self.router.score(p, task), 1)) for p in ranked
        ]
        attempts: list[Attempt] = []
        for provider in ranked[: self.max_attempts]:
            request = AIRequest(
                messages=[self._system(provider, style, private_context), *messages], task=task
            )
            started = time.perf_counter()
            try:
                self.manager.cost_guard.check_before_call(provider.info)
                response = await provider.generate(request)
                self.manager.cost_guard.check_after_call(provider.info, response)
            except ProviderError as exc:
                self.manager.record_failure(provider.id, exc)
                attempts.append(
                    Attempt(
                        agent_id=provider.id,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                )
                continue
            self.manager.record_success(provider.id, response)
            attempts.append(Attempt(agent_id=provider.id, ok=True, latency_ms=response.latency_ms))
            return OrchestratorResult(
                response=response, task=task, attempts=attempts, ranking=ranking
            )

        if not ranked:
            raise AllAgentsFailedError("no free agent is available right now", attempts)
        raise AllAgentsFailedError(f"all {len(attempts)} agents failed", attempts)

    async def ask(
        self,
        conversation: Conversation,
        text: str,
        *,
        style: str | None = None,
        private_context: str | None = None,
    ) -> OrchestratorResult:
        """Answer ``text`` inside ``conversation``. ``style`` adds per-channel instructions."""
        text = text.strip()
        if not text:
            raise ValueError("empty message")
        conversation.add_user(text)
        try:
            result = await self.complete(
                conversation.context(),
                classify(text),
                style=style,
                private_context=private_context,
            )
        except AllAgentsFailedError:
            # Nobody answered: forget the message so a retry does not duplicate it.
            conversation.drop_last_user()
            raise
        conversation.add_assistant(result.response.text, result.response.agent_id)
        return result
