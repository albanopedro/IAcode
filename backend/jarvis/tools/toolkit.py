"""The tool loop: ask an agent, run the tool it requests, give it the result, repeat.

    agent → ```tool {...}``` → executor (policy, confirmation, sandbox, audit)
          → result as untrusted data → agent → … → final answer in text

Only the final answer is stored in the conversation; the intermediate steps live
in this request only (and in the audit log).
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.core.orchestrator import Orchestrator
from jarvis.core.types import Message, OrchestratorResult, TaskType, ToolUse
from jarvis.tools.base import ToolContext
from jarvis.tools.executor import ToolExecutor
from jarvis.tools.protocol import FINAL_ANSWER_ONLY, format_result, parse_tool_call, tools_prompt

GAVE_UP = (
    "Não consegui concluir esta tarefa com as ferramentas disponíveis. "
    "Pode reformular o pedido ou me dar mais detalhes?"
)


@dataclass
class ToolKit:
    executor: ToolExecutor
    max_steps: int = 4

    def close(self) -> None:
        if self.executor.audit is not None:
            self.executor.audit.close()

    def prompt(self) -> str | None:
        tools = self.executor.usable()
        return tools_prompt(tools, self.max_steps) if tools else None

    async def run(
        self,
        orchestrator: Orchestrator,
        context: list[Message],
        task: TaskType,
        *,
        style: str | None,
        private_context: str | None,
        ctx: ToolContext,
        local_only: bool = False,
    ) -> OrchestratorResult:
        messages = list(context)
        used: list[ToolUse] = []
        tools_text = self.prompt()
        for step in range(self.max_steps + 1):
            last = step == self.max_steps or tools_text is None
            extra = FINAL_ANSWER_ONLY if (last and tools_text) else tools_text
            full_style = "\n\n".join(p for p in (style, extra) if p) or None
            result = await orchestrator.complete(
                messages,
                task,
                style=full_style,
                private_context=private_context,
                local_only=local_only,
            )
            call = parse_tool_call(result.response.text)
            if call is None:
                break
            if last:
                result.response.text = GAVE_UP
                break
            ctx.agent_id = result.response.agent_id
            outcome = await self.executor.execute(call, ctx)
            used.append(
                ToolUse(
                    name=call.name,
                    title=outcome.tool.title if outcome.tool else call.name,
                    decision=outcome.decision.value,
                    ok=outcome.ran and outcome.result.ok,
                )
            )
            untrusted = bool(outcome.tool and outcome.tool.untrusted_output)
            messages += [
                Message(role="assistant", content=result.response.text),
                Message(
                    role="user",
                    content=format_result(
                        call, outcome.result.text, ok=outcome.result.ok, untrusted=untrusted
                    ),
                ),
            ]
        result.tools = used
        return result
