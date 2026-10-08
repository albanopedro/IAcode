"""Router: ranks the available agents for a request.

Not a fixed A → B → C chain: every candidate gets a score, so an agent that is
better for the task wins even when another one is also available.

    score = priority
          + 60 × quality for the task        (config, or derived from capabilities)
          − 20 × (1 − recent success rate)   (last 20 calls)
          − 15 × consecutive failures
          −  2 × average latency in seconds
          + privacy bonus (local 8, zero-retention 5) + local bonus 5

Agents whose context window cannot hold the conversation are filtered out.
"""

from __future__ import annotations

from jarvis.core.agent_manager import AgentManager
from jarvis.core.provider import AIProvider
from jarvis.core.types import TASK_CAPABILITY, AIRequest, Privacy, TaskType

QUALITY_WEIGHT = 60.0
CAPABLE_QUALITY = 0.6
FALLBACK_QUALITY = 0.2
SUCCESS_WEIGHT = 20.0
FAILURE_PENALTY = 15.0
LATENCY_PENALTY_PER_SECOND = 2.0
PRIVACY_BONUS = {Privacy.LOCAL: 8.0, Privacy.ZERO_RETENTION: 5.0}
LOCAL_BONUS = 5.0
CHARS_PER_TOKEN = 3.5  # rough estimate; errs on the side of more tokens
OUTPUT_RESERVE = 1024  # tokens kept free for the answer


def estimate_tokens(request: AIRequest) -> int:
    chars = sum(len(m.content) for m in request.messages)
    return int(chars / CHARS_PER_TOKEN) + OUTPUT_RESERVE


class Router:
    def __init__(self, manager: AgentManager) -> None:
        self.manager = manager

    @staticmethod
    def quality(provider: AIProvider, task: TaskType) -> float:
        configured = provider.info.quality.get(task)
        if configured is not None:
            return max(0.0, min(float(configured), 1.0))
        return CAPABLE_QUALITY if provider.supports(TASK_CAPABILITY[task]) else FALLBACK_QUALITY

    def explain(self, provider: AIProvider, task: TaskType) -> dict[str, float]:
        """Score components, for debugging and for the UI."""
        info = provider.info
        state = self.manager.state(provider.id)
        parts = {
            "priority": float(info.priority),
            "quality": QUALITY_WEIGHT * self.quality(provider, task),
            "success": 0.0,
            "failures": -FAILURE_PENALTY * state.consecutive_failures,
            "latency": 0.0,
            "privacy": PRIVACY_BONUS.get(info.privacy, 0.0) + (LOCAL_BONUS if info.is_local else 0),
        }
        if state.success_rate is not None:
            parts["success"] = -SUCCESS_WEIGHT * (1 - state.success_rate)
        if state.avg_latency_ms is not None:
            parts["latency"] = -LATENCY_PENALTY_PER_SECOND * state.avg_latency_ms / 1000
        return parts

    def score(self, provider: AIProvider, task: TaskType) -> float:
        return sum(self.explain(provider, task).values())

    @staticmethod
    def fits(provider: AIProvider, request: AIRequest) -> bool:
        window = provider.context_window
        return window is None or estimate_tokens(request) <= window

    def rank(
        self, candidates: list[AIProvider], task: TaskType, request: AIRequest | None = None
    ) -> list[AIProvider]:
        if request is not None:
            candidates = [p for p in candidates if self.fits(p, request)]
        # Stable tie-break on id so the order is deterministic.
        return sorted(candidates, key=lambda p: (-self.score(p, task), p.id))
