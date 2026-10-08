"""AI Orchestrator: the brain that turns a user message into an answer.

user text → classify → rank free agents → try the best one
          → on failure, record it and try the next → answer (or a clear error)
"""

from __future__ import annotations

import time

from jarvis.core.agent_manager import AgentManager
from jarvis.core.classifier import classify
from jarvis.core.conversation import Conversation
from jarvis.core.errors import AllAgentsFailedError, ProviderError
from jarvis.core.router import Router
from jarvis.core.types import AIRequest, Attempt, Message, OrchestratorResult

SYSTEM_PROMPT = (
    "Você é o JARVIS, um assistente pessoal inteligente, educado e direto. "
    "Responda no idioma do usuário (português do Brasil por padrão), de forma clara "
    "e concisa, pois suas respostas também poderão ser faladas em voz alta. "
    "Você não tem acesso a arquivos, comandos ou internet nesta conversa."
)


class Orchestrator:
    def __init__(
        self,
        manager: AgentManager,
        router: Router | None = None,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_attempts: int = 5,
    ) -> None:
        self.manager = manager
        self.router = router or Router(manager)
        self.system_prompt = system_prompt
        self.max_attempts = max_attempts

    async def ask(self, conversation: Conversation, text: str) -> OrchestratorResult:
        text = text.strip()
        if not text:
            raise ValueError("empty message")

        await self.manager.refresh_health()
        conversation.add_user(text)
        task = classify(text)
        request = AIRequest(
            messages=[Message(role="system", content=self.system_prompt), *conversation.window()],
            task=task,
        )

        ranked = self.router.rank(self.manager.candidates(request.required), task)
        attempts: list[Attempt] = []
        for provider in ranked[: self.max_attempts]:
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
            conversation.add_assistant(response.text, provider.id)
            return OrchestratorResult(response=response, task=task, attempts=attempts)

        # Nobody answered: forget the message so a retry does not duplicate it.
        conversation.drop_last_user()
        if not ranked:
            raise AllAgentsFailedError("no free agent is available right now", attempts)
        raise AllAgentsFailedError(f"all {len(attempts)} agents failed", attempts)
