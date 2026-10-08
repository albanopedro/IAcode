"""Text CLI: check agents and chat with JARVIS in the terminal.

python -m jarvis status [--json]   # health of every agent (spends no quota)
python -m jarvis ask "..."         # one question
python -m jarvis chat              # conversation; /status, /limpar, /sair
python -m jarvis unblock <id>      # lift a persisted cost/billing block
python -m jarvis voice             # talk by voice (needs the [voice] extra)
python -m jarvis speak "..."       # test the voice
python -m jarvis transcribe a.wav  # test speech-to-text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime

from jarvis.agents.factory import build_agents
from jarvis.config import load_settings
from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.errors import AllAgentsFailedError
from jarvis.core.health_monitor import HealthMonitor
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.types import AgentStatus, Health, OrchestratorResult
from jarvis.core.usage_store import UsageStore

ICONS = {
    Health.AVAILABLE: "🟢",
    Health.COOLDOWN: "🟡",
    Health.OFFLINE: "🔴",
    Health.UNCONFIGURED: "⚪",
    Health.BLOCKED: "⛔",
    Health.UNKNOWN: "❔",
}


def build() -> tuple[AgentManager, Orchestrator, float]:
    settings = load_settings()
    store = UsageStore(settings.data_dir / "jarvis.db")
    manager = AgentManager(build_agents(settings), CostGuard(settings.cost_mode), store=store)
    return manager, Orchestrator(manager), settings.health_interval


def format_status(status: AgentStatus) -> str:
    icon = ICONS[status.health]
    line = f"{icon} {status.id:<32} {status.health.value:<13} prio {status.priority:>3}"
    extras = []
    if status.cooldown_until:
        seconds = (status.cooldown_until - datetime.now(UTC)).total_seconds()
        extras.append(f"cooldown {max(seconds, 0):.0f}s")
    if status.remaining_usage is not None:
        origin = "provedor" if status.remaining_source == "provider" else "hoje"
        total = f"/{status.rate_limit}" if status.rate_limit else ""
        extras.append(f"restam {status.remaining_usage}{total} ({origin})")
    if status.rpm_limit:
        extras.append(f"{status.requests_last_minute}/{status.rpm_limit} por min")
    if status.success_rate is not None:
        extras.append(f"sucesso {status.success_rate:.0%}")
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
    if result.ranking:
        top = " · ".join(f"{r.agent_id} {r.score:g}" for r in result.ranking[:4])
        print(f"    ranking: {top}")


def print_failure(error: AllAgentsFailedError) -> None:
    print(f"\nJARVIS: desculpe, nenhum agente gratuito conseguiu responder ({error}).")
    for attempt in error.attempts:
        print(f"    ✗ {attempt.agent_id}: {attempt.error}")


async def cmd_status(as_json: bool = False) -> int:
    manager, _, _ = build()
    try:
        await manager.refresh_health(force=True)
        if as_json:
            payload = {
                "cost_mode": manager.cost_guard.mode.value,
                "agents": [s.model_dump(mode="json") for s in manager.statuses()],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_statuses(manager)
    finally:
        await manager.close()
    return 0


async def cmd_unblock(agent_id: str) -> int:
    manager, _, _ = build()
    try:
        if agent_id not in {p.id for p in manager.providers}:
            print(f"agente desconhecido: {agent_id}")
            return 2
        if manager.unblock(agent_id):
            print(f"{agent_id} desbloqueado. Ele será verificado de novo antes do próximo uso.")
            return 0
        print(f"{agent_id} não estava bloqueado, ou é pago (bloqueado pelo COST_MODE).")
        return 1
    finally:
        await manager.close()


async def cmd_ask(question: str) -> int:
    manager, orchestrator, _ = build()
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
    manager, orchestrator, interval = build()
    conversation = Conversation()
    monitor = HealthMonitor(manager, interval)
    monitor.start()
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
        await monitor.stop()
        await manager.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS — only free AI agents")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show the health of every agent (spends no quota)")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    sub.add_parser("chat", help="start a conversation")
    unblock = sub.add_parser("unblock", help="lift a persisted cost/billing block")
    unblock.add_argument("agent_id")
    voice = sub.add_parser("voice", help="talk to JARVIS (microphone + speaker)")
    voice.add_argument("--tts", choices=["say", "piper"], help="override the configured voice")
    voice.add_argument("--once", action="store_true", help="answer one question and stop")
    speak = sub.add_parser("speak", help="say a text with the configured voice")
    speak.add_argument("text", nargs="+")
    speak.add_argument("--tts", choices=["say", "piper"])
    transcribe = sub.add_parser("transcribe", help="transcribe a 16-bit WAV file")
    transcribe.add_argument("path")
    args = parser.parse_args(argv)

    if args.command in ("voice", "speak", "transcribe"):
        try:
            from jarvis.voice import cli as voice_cli
        except ImportError as exc:
            print(f"voz indisponível ({exc}). Instale com: pip install -e '.[voice]'")
            return 2
        if args.command == "voice":
            return asyncio.run(voice_cli.cmd_voice(args.tts, args.once))
        if args.command == "speak":
            return asyncio.run(voice_cli.cmd_speak(" ".join(args.text), args.tts))
        return asyncio.run(voice_cli.cmd_transcribe(args.path))

    if args.command == "status":
        return asyncio.run(cmd_status(args.json))
    if args.command == "unblock":
        return asyncio.run(cmd_unblock(args.agent_id))
    if args.command == "ask":
        return asyncio.run(cmd_ask(" ".join(args.question)))
    return asyncio.run(cmd_chat())


if __name__ == "__main__":
    sys.exit(main())
