"""Text CLI for Phase 2: check agents and chat with JARVIS in the terminal.

python -m jarvis status        # health of every agent (spends no quota)
python -m jarvis ask "..."     # one question
python -m jarvis chat          # conversation; /status, /limpar, /sair
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from jarvis.agents.factory import build_agents
from jarvis.config import load_settings
from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import AllAgentsFailedError
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.types import AgentStatus, Health, OrchestratorResult

ICONS = {
    Health.AVAILABLE: "🟢",
    Health.COOLDOWN: "🟡",
    Health.OFFLINE: "🔴",
    Health.UNCONFIGURED: "⚪",
    Health.BLOCKED: "⛔",
    Health.UNKNOWN: "❔",
}


def build() -> tuple[AgentManager, Orchestrator]:
    settings = load_settings()
    manager = AgentManager(build_agents(settings), CostGuard(settings.cost_mode))
    return manager, Orchestrator(manager)


def format_status(status: AgentStatus) -> str:
    icon = ICONS[status.health]
    line = f"{icon} {status.id:<32} {status.health.value:<13} prio {status.priority:>3}"
    extras = []
    if status.cooldown_until:
        seconds = (status.cooldown_until - datetime.now(UTC)).total_seconds()
        extras.append(f"cooldown {max(seconds, 0):.0f}s")
    if status.remaining_usage is not None:
        extras.append(f"restam {status.remaining_usage}/{status.rate_limit} hoje")
    if status.avg_latency_ms is not None:
        extras.append(f"{status.avg_latency_ms / 1000:.1f}s")
    if status.last_error and status.health is not Health.AVAILABLE:
        extras.append(status.last_error[:90])
    return line + ("  · " + " · ".join(extras) if extras else "")


def print_statuses(manager: AgentManager) -> None:
    print(f"COST_MODE={manager.cost_guard.mode.value}")
    for status in manager.statuses():
        print(format_status(status))


def print_result(result: OrchestratorResult) -> None:
    print(f"\nJARVIS: {result.response.text}\n")
    trail = " → ".join(("✓ " if a.ok else "✗ ") + a.agent_id for a in result.attempts)
    cost = result.response.cost
    print(
        f"  [{result.task.value}] {trail} · {result.response.latency_ms / 1000:.1f}s"
        f" · custo US$ {cost if cost is not None else 0:g}"
    )
    for attempt in result.attempts:
        if not attempt.ok:
            print(f"    ✗ {attempt.agent_id}: {attempt.error}")


def print_failure(error: AllAgentsFailedError) -> None:
    print(f"\nJARVIS: desculpe, nenhum agente gratuito conseguiu responder ({error}).")
    for attempt in error.attempts:
        print(f"    ✗ {attempt.agent_id}: {attempt.error}")


async def cmd_status() -> int:
    manager, _ = build()
    try:
        await manager.refresh_health(force=True)
        print_statuses(manager)
    finally:
        await manager.close()
    return 0


async def cmd_ask(question: str) -> int:
    manager, orchestrator = build()
    try:
        result = await orchestrator.ask(Conversation(), question)
        print_result(result)
        return 0
    except AllAgentsFailedError as exc:
        print_failure(exc)
        return 1
    finally:
        await manager.close()


async def cmd_chat() -> int:
    manager, orchestrator = build()
    conversation = Conversation()
    print("JARVIS pronto. Comandos: /status, /limpar, /sair\n")
    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, "Você: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in ("/sair", "/exit", "/quit"):
                break
            if text == "/status":
                await manager.refresh_health(force=True)
                print_statuses(manager)
                continue
            if text == "/limpar":
                conversation.clear()
                print("(conversa apagada)")
                continue
            try:
                print_result(await orchestrator.ask(conversation, text))
            except AllAgentsFailedError as exc:
                print_failure(exc)
    finally:
        await manager.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS — only free AI agents")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the health of every agent (spends no quota)")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    sub.add_parser("chat", help="start a conversation")
    args = parser.parse_args(argv)

    if args.command == "status":
        return asyncio.run(cmd_status())
    if args.command == "ask":
        return asyncio.run(cmd_ask(" ".join(args.question)))
    return asyncio.run(cmd_chat())


if __name__ == "__main__":
    sys.exit(main())
