"""Cost guard: the single place that decides whether an agent may be called.

Rule (COST_MODE=FREE_ONLY, the default):
    local / free / free_with_limits  → allowed
    paid                             → BLOCKED
    any reported cost > 0            → BLOCKED (the agent is disabled for good)

A paid agent can only run when BOTH the global mode is ALLOW_PAID and the agent
itself opts in with ``allow_paid = true``. Neither is ever on by default.
"""

from __future__ import annotations

from jarvis.core.errors import CostViolationError
from jarvis.core.types import AgentInfo, AIResponse, CostClass, CostMode

FREE_CLASSES = frozenset({CostClass.LOCAL, CostClass.FREE, CostClass.FREE_WITH_LIMITS})


class CostGuard:
    def __init__(self, mode: CostMode = CostMode.FREE_ONLY) -> None:
        self.mode = mode

    def allowed(self, info: AgentInfo) -> tuple[bool, str]:
        if info.cost_class in FREE_CLASSES:
            return True, f"{info.cost_class.value}: allowed"
        if self.mode is CostMode.ALLOW_PAID and info.allow_paid:
            return True, "paid agent explicitly allowed (COST_MODE=ALLOW_PAID + allow_paid)"
        return False, f"paid agent blocked by COST_MODE={self.mode.value}"

    def check_before_call(self, info: AgentInfo) -> None:
        ok, reason = self.allowed(info)
        if not ok:
            raise CostViolationError(f"{info.id}: {reason}")

    def check_after_call(self, info: AgentInfo, response: AIResponse) -> None:
        if response.cost is None or response.cost <= 0:
            return
        if self.mode is CostMode.ALLOW_PAID and info.allow_paid:
            return
        raise CostViolationError(
            f"{info.id}: provider reported a cost of US$ {response.cost} — agent blocked"
        )
