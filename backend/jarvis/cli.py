"""Text CLI: check agents and chat with JARVIS in the terminal.

python -m jarvis status [--json]   # health of every agent (spends no quota)
python -m jarvis ask "..."         # one question
python -m jarvis chat              # conversation; /status, /limpar, /sair
python -m jarvis unblock <id>      # lift a persisted cost/billing block
python -m jarvis voice             # talk by voice (needs the [voice] extra)
python -m jarvis speak "..."       # test the voice
python -m jarvis transcribe a.wav  # test speech-to-text
python -m jarvis serve             # web interface API on http://127.0.0.1:8300
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
from jarvis.core.types import AgentStatus, Health, OrchestratorResult
from jarvis.core.usage_store import UsageStore
from jarvis.memory.assistant import Assistant
from jarvis.memory.factory import build_assistant

ICONS = {
    Health.AVAILABLE: "🟢",
    Health.COOLDOWN: "🟡",
    Health.OFFLINE: "🔴",
    Health.UNCONFIGURED: "⚪",
    Health.BLOCKED: "⛔",
    Health.UNKNOWN: "❔",
}


def build() -> tuple[AgentManager, Assistant, float]:
    settings = load_settings()
    store = UsageStore(settings.data_dir / "jarvis.db")
    manager = AgentManager(build_agents(settings), CostGuard(settings.cost_mode), store=store)
    return manager, build_assistant(settings, manager), settings.health_interval


def open_conversation(assistant: Assistant, resume: bool = False) -> Conversation:
    """A saved conversation when memory is on (the last one with ``resume``)."""
    if assistant.memory is None:
        return Conversation()
    if resume and (latest := assistant.memory.latest_conversation()) is not None:
        return latest
    return assistant.memory.new_conversation()


async def shutdown(manager: AgentManager, assistant: Assistant) -> None:
    await assistant.wait_background()  # let a pending summary finish
    await manager.close()
    await assistant.close()


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
    for use in result.tools:
        mark = "✓" if use.ok else "✗"
        print(f"    🔧 {mark} {use.title} ({use.decision})")


async def cli_confirm(call, tool, reason: str) -> bool:
    """Ask in the terminal before a tool that needs permission runs."""
    print(f"\n⚠️  O JARVIS quer usar: {tool.describe_call(call.args)}")
    print(f"   Motivo da confirmação: {reason}")
    try:
        answer = await asyncio.to_thread(input, "   Permitir? [s/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("s", "sim", "y", "yes")


def cli_tool_context():
    from jarvis.tools.base import ToolContext

    def on_event(kind: str, data: dict) -> None:
        if kind == "tool_start":
            print(f"   🔧 usando {data['title']}…", flush=True)

    return ToolContext(confirm=cli_confirm, on_event=on_event)


def print_failure(error: AllAgentsFailedError) -> None:
    print(f"\nJARVIS: desculpe, nenhum agente gratuito conseguiu responder ({error}).")
    for attempt in error.attempts:
        print(f"    ✗ {attempt.agent_id}: {attempt.error}")


async def cmd_status(as_json: bool = False) -> int:
    manager, assistant, _ = build()
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
        await shutdown(manager, assistant)
    return 0


async def cmd_unblock(agent_id: str) -> int:
    manager, assistant, _ = build()
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
        await shutdown(manager, assistant)


async def cmd_ask(question: str, local_only: bool = False) -> int:
    manager, assistant, _ = build()
    try:
        result = await assistant.ask(
            open_conversation(assistant),
            question,
            tool_context=cli_tool_context(),
            local_only=local_only,
        )
        print_result(result)
        return 0
    except AllAgentsFailedError as exc:
        print_failure(exc)
        return 1
    finally:
        await shutdown(manager, assistant)


async def cmd_chat(resume: bool = False, local_only: bool = False) -> int:
    manager, assistant, interval = build()
    conversation = open_conversation(assistant, resume)
    monitor = HealthMonitor(manager, interval)
    monitor.start()
    if conversation.turns:
        print(f"Continuando: {conversation.title} ({len(conversation.turns)} mensagens)")
    if local_only:
        print("🔒 Modo só local: nada sai deste Mac (só o modelo local responde).")
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
                conversation = open_conversation(assistant)
                print("(nova conversa; a anterior continua no histórico)")
                continue
            try:
                print_result(
                    await assistant.ask(
                        conversation,
                        text,
                        tool_context=cli_tool_context(),
                        local_only=local_only,
                    )
                )
            except AllAgentsFailedError as exc:
                print_failure(exc)
    finally:
        await monitor.stop()
        await shutdown(manager, assistant)
    return 0


def _memory_store():
    from jarvis.memory.store import MemoryStore

    return MemoryStore(load_settings().data_dir / "memory.db")


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} Digite SIM para confirmar: ").strip() == "SIM"
    except EOFError:
        return False


def cmd_memory(action: str, value: str | None) -> int:
    store = _memory_store()
    try:
        if action == "list":
            facts = store.facts()
            if not facts:
                print("(nenhuma lembrança guardada)")
            for fact in facts:
                print(f"{fact.id:>4}  {fact.text}")
        elif action == "add":
            saved = store.add_fact(value or "")
            print(f"guardado: {saved.text}" if saved else "já estava guardado (ou vazio)")
        elif action == "forget":
            ok = value is not None and value.isdigit() and store.delete_fact(int(value))
            print("esquecido" if ok else "não encontrei esse número (veja: jarvis memory list)")
            return 0 if ok else 1
        elif action == "clear":
            if not _confirm("Apagar TODAS as lembranças de longo prazo?"):
                print("cancelado")
                return 1
            print(f"{store.delete_all_facts()} lembranças apagadas")
        return 0
    finally:
        store.close()


def cmd_local(action: str) -> int:
    from jarvis.agents.local.mlx import LocalModelStore, mlx_available

    settings = load_settings()
    repo = settings.local.model
    store = LocalModelStore(settings.data_dir / "models" / "mlx")
    if action == "status":
        print(f"modelo: {repo}")
        print(f"MLX: {mlx_available() or 'ok'}")
        if store.is_downloaded(repo):
            print(f"baixado: sim ({store.size_bytes(repo) / 1e9:.2f} GB em {store.path(repo)})")
        else:
            print("baixado: não (rode: jarvis local download)")
        print(f"ativo no JARVIS: {'sim' if settings.local.enabled else 'não ([local] enabled)'}")
        return 0
    if action == "download":
        if (reason := mlx_available()) is not None:
            print(reason)
            return 2
        if store.is_downloaded(repo):
            print("o modelo já está baixado")
            return 0
        print(f"Baixando {repo} do Hugging Face (gratuito, sem conta)…")
        path = store.download(repo)
        print(f"pronto: {store.size_bytes(repo) / 1e9:.2f} GB em {path}")
        return 0
    if not _confirm(f"Apagar o modelo local {repo} do disco?"):
        print("cancelado")
        return 1
    print("apagado" if store.remove(repo) else "não estava baixado")
    return 0


def cmd_tools(action: str) -> int:
    from jarvis.tools.audit import AuditLog
    from jarvis.tools.base import Policy
    from jarvis.tools.executor import ToolExecutor
    from jarvis.tools.factory import all_tools

    settings = load_settings()
    if action == "list":
        policies = {k: Policy(v) for k, v in settings.tools.policies.items()}
        store = _memory_store()
        executor = ToolExecutor(all_tools(settings, store), policies=policies)
        store.close()
        print(
            f"ferramentas {'ligadas' if settings.tools.enabled else 'DESLIGADAS'}"
            f" · até {settings.tools.max_steps} por pergunta"
        )
        for tool in executor.tools.values():
            missing = tool.available()
            state = f"indisponível: {missing}" if missing else executor.policy(tool).value
            print(f"  {tool.name:<14} {tool.title:<26} risco {tool.risk.name.lower():<9} {state}")
        return 0
    log = AuditLog(settings.data_dir / "audit.db")
    try:
        entries = log.recent(30)
        if not entries:
            print("(nenhuma ferramenta usada ainda)")
        for e in reversed(entries):
            when = datetime.fromtimestamp(e.at).strftime("%d/%m %H:%M:%S")
            status = "" if e.ok is None else (" ✓" if e.ok else " ✗")
            print(f"{when}  {e.tool:<14} {e.decision:<17}{status}  {e.args[:80]}")
        return 0
    finally:
        log.close()


def cmd_history(action: str, value: str | None) -> int:
    store = _memory_store()
    try:
        if action == "list":
            for info in store.list_conversations():
                when = datetime.fromtimestamp(info.updated_at).strftime("%d/%m %H:%M")
                print(f"{info.id[:8]}  {when}  {info.messages:>3} msgs  {info.title}")
        elif action == "show":
            matches = [c for c in store.list_conversations(500) if c.id.startswith(value or "-")]
            if len(matches) != 1:
                print("informe o início do id (veja: jarvis history list)")
                return 1
            conversation = store.load_conversation(matches[0].id)
            for turn in conversation.turns:
                who = "Você" if turn.message.role == "user" else "JARVIS"
                print(f"{who}: {turn.message.content}\n")
        elif action == "clear":
            if not _confirm("Apagar TODO o histórico de conversas?"):
                print("cancelado")
                return 1
            print(f"{store.delete_all_conversations()} conversas apagadas")
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="JARVIS — only free AI agents")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show the health of every agent (spends no quota)")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--local", action="store_true", help="only the local model (nothing leaves)")
    chat = sub.add_parser("chat", help="start a conversation")
    chat.add_argument("--continue", dest="resume", action="store_true", help="resume the last one")
    chat.add_argument("--local", action="store_true", help="only the local model (nothing leaves)")
    local = sub.add_parser("local", help="offline model: status | download | remove")
    local.add_argument("action", choices=["status", "download", "remove"])
    memory = sub.add_parser(
        "memory", help="long-term memory: list | add <text> | forget <id> | clear"
    )
    memory.add_argument("action", choices=["list", "add", "forget", "clear"])
    memory.add_argument("value", nargs="*")
    tools = sub.add_parser("tools", help="tools: list (policies) | log (audit)")
    tools.add_argument("action", choices=["list", "log"])
    history = sub.add_parser("history", help="saved conversations: list | show <id> | clear")
    history.add_argument("action", choices=["list", "show", "clear"])
    history.add_argument("value", nargs="?")
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
    serve = sub.add_parser("serve", help="run the web interface server (127.0.0.1 only)")
    serve.add_argument("--port", type=int, default=8300)
    serve.add_argument("--open", action="store_true", help="open the interface in the browser")
    sub.add_parser("doctor", help="check what is installed, configured and downloaded")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from jarvis.doctor import Level, render, run_checks

        checks = run_checks(load_settings())
        print(render(checks))
        return 1 if any(c.level is Level.FAIL for c in checks) else 0

    if args.command == "serve":
        try:
            import uvicorn

            from jarvis.server.app import create_app, default_origins
        except ImportError as exc:
            print(f"servidor indisponível ({exc}). Instale com: pip install -e '.[server]'")
            return 2
        app = create_app(allowed_origins=default_origins(args.port))
        url = f"http://127.0.0.1:{args.port}"
        print(f"JARVIS em {url}  (Ctrl+C para encerrar)")
        if args.open:
            import threading
            import webbrowser

            threading.Timer(1.5, webbrowser.open, [url]).start()
        # Never 0.0.0.0: JARVIS must not be reachable from the network.
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
        return 0

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

    if args.command == "memory":
        return cmd_memory(args.action, " ".join(args.value) or None)
    if args.command == "history":
        return cmd_history(args.action, args.value)
    if args.command == "tools":
        return cmd_tools(args.action)
    if args.command == "local":
        return cmd_local(args.action)
    if args.command == "status":
        return asyncio.run(cmd_status(args.json))
    if args.command == "unblock":
        return asyncio.run(cmd_unblock(args.agent_id))
    if args.command == "ask":
        return asyncio.run(cmd_ask(" ".join(args.question), args.local))
    return asyncio.run(cmd_chat(args.resume, args.local))


if __name__ == "__main__":
    sys.exit(main())
