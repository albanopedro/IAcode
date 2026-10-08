"""Router: ranks the available agents for a task.

Not a fixed A → B → C chain: every candidate gets a score, so an agent that is
better for the task wins even when another one is also available.
"""

from __future__ import annotations

from jarvis.core.agent_manager import AgentManager
from jarvis.core.provider import AIProvider
from jarvis.core.types import TASK_CAPABILITY, Privacy, TaskType

CAPABILITY_BONUS = 30.0
FAILURE_PENALTY = 15.0
LATENCY_PENALTY_PER_SECOND = 2.0
PRIVACY_BONUS = {Privacy.LOCAL: 8.0, Privacy.ZERO_RETENTION: 5.0}
LOCAL_BONUS = 5.0


class Router:
    def __init__(self, manager: AgentManager) -> None:
        self.manager = manager

    def score(self, provider: AIProvider, task: TaskType) -> float:
        info = provider.info
        state = self.manager.state(provider.id)
        score = float(info.priority)
        if provider.supports(TASK_CAPABILITY[task]):
            score += CAPABILITY_BONUS
        score -= FAILURE_PENALTY * state.consecutive_failures
        if state.avg_latency_ms is not None:
            score -= LATENCY_PENALTY_PER_SECOND * state.avg_latency_ms / 1000
        score += PRIVACY_BONUS.get(info.privacy, 0.0)
        if info.is_local:
            score += LOCAL_BONUS
        return score

    def rank(self, candidates: list[AIProvider], task: TaskType) -> list[AIProvider]:
        # Stable tie-break on id so the order is deterministic.
        return sorted(candidates, key=lambda p: (-self.score(p, task), p.id))
