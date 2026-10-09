"""The only door between agents and tools.

For every requested call:
1. the tool must exist, be enabled and be available (e.g. folders configured);
2. the policy decides: allow / confirm / deny (the tool's default, or your
   override in config/agents.toml — which never bypasses the checks below);
3. after untrusted content (a web page, a file) entered the request, every
   SENSITIVE or DANGEROUS tool needs your confirmation, even if it is set to
   "allow" — this is the defence against prompt injection;
4. DANGEROUS tools always need confirmation, whatever the config says;
5. the call runs with a timeout, its output is capped, and it is audited.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from jarvis.tools.audit import AuditLog
from jarvis.tools.base import Policy, Risk, Tool, ToolCall, ToolContext, ToolError, ToolResult

MAX_OUTPUT_CHARS = 8000


class Decision(StrEnum):
    ALLOWED = "allowed"
    CONFIRMED = "confirmed"
    DENIED_BY_USER = "denied_by_user"
    DENIED_BY_POLICY = "denied_by_policy"
    UNAVAILABLE = "unavailable"
    UNKNOWN_TOOL = "unknown_tool"
    RATE_LIMITED = "rate_limited"


@dataclass
class ToolOutcome:
    call: ToolCall
    decision: Decision
    result: ToolResult
    tool: Tool | None = None
    duration_ms: float = 0.0

    @property
    def ran(self) -> bool:
        return self.decision in (Decision.ALLOWED, Decision.CONFIRMED)


class ToolExecutor:
    def __init__(
        self,
        tools: list[Tool],
        *,
        policies: dict[str, Policy] | None = None,
        audit: AuditLog | None = None,
        max_calls_per_minute: int = 20,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.policies = policies or {}
        self.audit = audit
        self.max_calls_per_minute = max_calls_per_minute
        self._recent: deque[float] = deque()

    def policy(self, tool: Tool) -> Policy:
        policy = self.policies.get(tool.name, tool.default_policy)
        if tool.risk is Risk.DANGEROUS and policy is Policy.ALLOW:
            return Policy.CONFIRM  # never run code without asking
        return policy

    def usable(self) -> list[Tool]:
        """Tools worth describing to the agent (not denied, currently available)."""
        return [
            t
            for t in self.tools.values()
            if self.policy(t) is not Policy.DENY and t.available() is None
        ]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolOutcome:
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self.tools)) or "nenhuma"
            return self._refuse(
                call,
                None,
                ctx,
                Decision.UNKNOWN_TOOL,
                f"ferramenta desconhecida: {call.name}. Disponíveis: {known}",
            )
        if (reason := tool.available()) is not None:
            return self._refuse(call, tool, ctx, Decision.UNAVAILABLE, reason)
        policy = self.policy(tool)
        if policy is Policy.DENY:
            return self._refuse(
                call,
                tool,
                ctx,
                Decision.DENIED_BY_POLICY,
                "esta ferramenta está desativada pelo usuário",
            )
        if not self._within_rate():
            return self._refuse(
                call,
                tool,
                ctx,
                Decision.RATE_LIMITED,
                "limite de chamadas de ferramentas por minuto atingido",
            )

        decision = Decision.ALLOWED
        needs_confirm = policy is Policy.CONFIRM or (ctx.tainted and tool.risk >= Risk.SENSITIVE)
        if needs_confirm:
            reason = self._confirm_reason(tool, policy, ctx)
            ctx.emit(
                "tool_confirm",
                {
                    "tool": tool.name,
                    "title": tool.title,
                    "call": tool.describe_call(call.args),
                    "reason": reason,
                },
            )
            approved = ctx.confirm is not None and await self._ask(ctx, call, tool, reason)
            if not approved:
                message = (
                    "o usuário NÃO autorizou esta ação. Não tente de novo; explique o que "
                    "precisaria fazer."
                    if ctx.confirm is not None
                    else "esta ação precisa de confirmação na tela e não pôde ser confirmada aqui"
                )
                return self._refuse(call, tool, ctx, Decision.DENIED_BY_USER, message)
            decision = Decision.CONFIRMED

        ctx.calls += 1
        ctx.emit(
            "tool_start",
            {"tool": tool.name, "title": tool.title, "call": tool.describe_call(call.args)},
        )
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.run(call.args, ctx), tool.timeout)
        except ToolError as exc:
            result = ToolResult(str(exc), ok=False)
        except TimeoutError:
            result = ToolResult(f"tempo esgotado ({tool.timeout:.0f}s)", ok=False)
        except Exception as exc:  # a broken tool must never break the conversation
            result = ToolResult(f"erro interno da ferramenta: {type(exc).__name__}", ok=False)
        duration = (time.perf_counter() - started) * 1000
        if len(result.text) > MAX_OUTPUT_CHARS:
            result.text = result.text[:MAX_OUTPUT_CHARS] + "\n[…cortado]"
        if tool.untrusted_output and result.ok:
            ctx.tainted = True
        self._log(call, ctx, decision, result, duration)
        ctx.emit("tool_end", {"tool": tool.name, "title": tool.title, "ok": result.ok})
        return ToolOutcome(call, decision, result, tool, duration)

    # -- helpers --------------------------------------------------------------------

    @staticmethod
    def _confirm_reason(tool: Tool, policy: Policy, ctx: ToolContext) -> str:
        if policy is Policy.CONFIRM:
            return {
                Risk.DANGEROUS: "executa código (dentro de uma sandbox)",
                Risk.SENSITIVE: "acessa seus dados ou a internet",
                Risk.SAFE: "configurada para pedir confirmação",
            }[tool.risk]
        return "pedida depois de ler conteúdo externo (proteção contra prompt injection)"

    @staticmethod
    async def _ask(ctx: ToolContext, call: ToolCall, tool: Tool, reason: str) -> bool:
        try:
            return bool(await ctx.confirm(call, tool, reason))
        except Exception:
            return False

    def _within_rate(self) -> bool:
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if len(self._recent) >= self.max_calls_per_minute:
            return False
        self._recent.append(now)
        return True

    def _refuse(self, call, tool, ctx, decision: Decision, message: str) -> ToolOutcome:
        result = ToolResult(message, ok=False)
        self._log(call, ctx, decision, result, 0.0)
        ctx.emit(
            "tool_end",
            {
                "tool": call.name,
                "title": tool.title if tool else call.name,
                "ok": False,
                "decision": decision.value,
            },
        )
        return ToolOutcome(call, decision, result, tool)

    def _log(self, call, ctx, decision: Decision, result: ToolResult, duration: float) -> None:
        if self.audit is not None:
            self.audit.record(
                tool=call.name,
                args=call.args,
                decision=decision.value,
                ok=result.ok if decision in (Decision.ALLOWED, Decision.CONFIRMED) else None,
                result=result.text,
                duration_ms=duration,
                conversation_id=ctx.conversation_id,
                agent_id=ctx.agent_id,
            )
