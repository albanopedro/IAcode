"""One browser connection: text chat, continuous voice and live events.

Client → server (JSON text messages, plus binary microphone audio):
    {"type": "text", "text": "...", "speak": false}   ask in writing (optionally hear it)
    {"type": "voice_start", "tts": "say" | "piper"}    start the continuous voice loop
    {"type": "voice_stop"}                             stop listening
    {"type": "interrupt"}                              stop JARVIS talking
    {"type": "clear"} / {"type": "new_conversation"}   start a new conversation
    {"type": "open_conversation", "id": "..."}         reopen a saved conversation
    {"type": "delete_conversation", "id": "..."}       delete a saved conversation
    {"type": "list_conversations"}                     saved conversations
    {"type": "list_facts"} / {"type": "forget_fact", "id": 1}
    {"type": "clear_facts", "confirm": true}           forget every long-term fact
    {"type": "status"}                                 ask for the agents' status
    <binary> PCM16 mono 16 kHz frames while the voice loop runs

Server → client (JSON), plus binary WAV messages with JARVIS's voice:
    hello · state · transcript · answer · agents · stop_audio · cleared · error
    history (the open conversation) · conversations · facts
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from jarvis.core.conversation import Conversation
from jarvis.core.errors import AllAgentsFailedError
from jarvis.core.types import OrchestratorResult
from jarvis.server.runtime import JarvisRuntime
from jarvis.server.ws_audio import QueueAudioSource, WebSocketSink
from jarvis.voice.session import Phase, VoiceSession

MAX_TEXT = 4000
MAX_JSON_BYTES = 16 * 1024


def answer_event(result: OrchestratorResult, mode: str) -> dict[str, Any]:
    response = result.response
    return {
        "type": "answer",
        "mode": mode,
        "text": response.text,
        "agent_id": response.agent_id,
        "model": response.model,
        "task": result.task.value,
        "latency_ms": round(response.latency_ms),
        "cost": response.cost or 0,
        "attempts": [a.model_dump() for a in result.attempts],
        "ranking": [r.model_dump() for r in result.ranking[:5]],
    }


def agents_event(runtime: JarvisRuntime) -> dict[str, Any]:
    return {
        "type": "agents",
        "cost_mode": runtime.manager.cost_guard.mode.value,
        "agents": [s.model_dump(mode="json") for s in runtime.manager.statuses()],
    }


def history_event(conversation: Conversation) -> dict[str, Any]:
    return {
        "type": "history",
        "conversation_id": conversation.id,
        "title": conversation.title,
        "has_summary": bool(conversation.summary),
        "messages": [
            {
                "role": turn.message.role,
                "text": turn.message.content,
                "agent_id": turn.agent_id,
                "created_at": turn.created_at,
            }
            for turn in conversation.turns
        ],
    }


class _LockedOrchestrator:
    """Text and voice share one conversation: never answer two messages at once."""

    def __init__(self, orchestrator, lock: asyncio.Lock) -> None:
        self._orchestrator = orchestrator
        self._lock = lock

    async def ask(self, conversation, text, *, style=None):
        async with self._lock:
            return await self._orchestrator.ask(conversation, text, style=style)


class Connection:
    def __init__(self, websocket: WebSocket, runtime: JarvisRuntime) -> None:
        self.ws = websocket
        self.runtime = runtime
        memory = runtime.memory
        self.conversation: Conversation = (
            (memory.latest_conversation() or memory.new_conversation())
            if memory
            else Conversation()
        )
        self.voice_session: VoiceSession | None = None
        self.source = QueueAudioSource()
        self.sink = WebSocketSink(self._send_audio)
        self.orchestrator = _LockedOrchestrator(runtime.assistant, asyncio.Lock())
        self.outbox: asyncio.Queue[dict | bytes] = asyncio.Queue()
        self.voice_task: asyncio.Task | None = None
        self.tasks: set[asyncio.Task] = set()

    # -- sending ------------------------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        self.outbox.put_nowait(event)

    def state(self, value: str) -> None:
        self.emit({"type": "state", "state": value})

    async def _send_audio(self, wav: bytes, duration: float) -> None:
        self.outbox.put_nowait(wav)

    async def _sender(self) -> None:
        while True:
            item = await self.outbox.get()
            if isinstance(item, bytes):
                await self.ws.send_bytes(item)
            else:
                await self.ws.send_json(item)

    # -- main loop ----------------------------------------------------------------

    async def run(self) -> None:
        sender = asyncio.create_task(self._sender())
        self.emit(
            {
                "type": "hello",
                "voice_ready": self.runtime.voice_ready,
                "tts_engine": self.runtime.settings.voice.tts_engine,
                "max_text": MAX_TEXT,
            }
        )
        self.emit(agents_event(self.runtime))
        self.emit(history_event(self.conversation))
        self._emit_memory()
        self.state("idle")
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if self.voice_task is not None:
                        self.source.push(message["bytes"])
                elif message.get("text") is not None:
                    await self.handle_text(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            for task in [self.voice_task, *self.tasks]:
                if task is not None:
                    task.cancel()
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def handle_text(self, raw: str) -> None:
        if len(raw) > MAX_JSON_BYTES:
            self.emit({"type": "error", "message": "mensagem grande demais"})
            return
        try:
            message = json.loads(raw)
            kind = message["type"]
        except (ValueError, KeyError, TypeError):
            self.emit({"type": "error", "message": "mensagem inválida"})
            return

        if kind == "text":
            text = message.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
                self.emit({"type": "error", "message": f"texto vazio ou maior que {MAX_TEXT}"})
                return
            self._spawn(self._ask_text(text.strip(), bool(message.get("speak"))))
        elif kind == "voice_start":
            if self.voice_task is None or self.voice_task.done():
                engine = message.get("tts")
                if engine not in (None, "say", "piper"):
                    self.emit({"type": "error", "message": "voz desconhecida"})
                    return
                self.voice_task = asyncio.create_task(self._voice_loop(engine))
        elif kind == "voice_stop":
            await self._stop_voice()
        elif kind == "interrupt":
            self.sink.stop()
            self.emit({"type": "stop_audio"})
        elif kind in ("clear", "new_conversation"):
            memory = self.runtime.memory
            self._switch(memory.new_conversation() if memory else Conversation())
            self.emit({"type": "cleared"})
        elif kind == "open_conversation":
            memory = self.runtime.memory
            found = memory.load_conversation(str(message.get("id"))) if memory else None
            if found is None:
                self.emit({"type": "error", "message": "conversa não encontrada"})
                return
            self._switch(found)
        elif kind == "delete_conversation":
            memory = self.runtime.memory
            target = str(message.get("id"))
            if memory and memory.delete_conversation(target):
                if target == self.conversation.id:
                    self._switch(memory.new_conversation())
                self._emit_memory()
        elif kind == "list_conversations" or kind == "list_facts":
            self._emit_memory()
        elif kind == "forget_fact":
            memory = self.runtime.memory
            fact_id = message.get("id")
            if memory and isinstance(fact_id, int) and memory.delete_fact(fact_id):
                self._emit_memory()
            else:
                self.emit({"type": "error", "message": "lembrança não encontrada"})
        elif kind == "clear_facts":
            memory = self.runtime.memory
            if memory and message.get("confirm") is True:
                memory.delete_all_facts()
                self._emit_memory()
        elif kind == "status":
            self.emit(agents_event(self.runtime))
        else:
            self.emit({"type": "error", "message": f"tipo desconhecido: {kind}"})

    # -- memory -------------------------------------------------------------------

    def _switch(self, conversation: Conversation) -> None:
        self.conversation = conversation
        if self.voice_session is not None:
            self.voice_session.conversation = conversation
        self.emit(history_event(conversation))
        self._emit_memory()

    def _emit_memory(self) -> None:
        memory = self.runtime.memory
        if memory is None:
            self.emit({"type": "facts", "enabled": False, "facts": []})
            self.emit({"type": "conversations", "conversations": []})
            return
        self.emit(
            {
                "type": "facts",
                "enabled": True,
                "facts": [
                    {"id": f.id, "text": f.text, "created_at": f.created_at} for f in memory.facts()
                ],
            }
        )
        self.emit(
            {
                "type": "conversations",
                "current": self.conversation.id,
                "conversations": [
                    {
                        "id": c.id,
                        "title": c.title,
                        "updated_at": c.updated_at,
                        "messages": c.messages,
                    }
                    for c in memory.list_conversations()
                ],
            }
        )

    def _after_answer(self, result: OrchestratorResult) -> None:
        self.emit(agents_event(self.runtime))
        if self.runtime.memory is not None:
            self._emit_memory()  # new title / new fact

    # -- text ---------------------------------------------------------------------

    async def _ask_text(self, text: str, speak: bool) -> None:
        self.state("thinking")
        try:
            result = await self.orchestrator.ask(self.conversation, text)
        except AllAgentsFailedError as exc:
            self.emit(
                {
                    "type": "error",
                    "message": f"nenhum agente gratuito conseguiu responder ({exc})",
                    "attempts": [a.model_dump() for a in exc.attempts],
                }
            )
            self._settle()
            return
        self.emit(answer_event(result, "text"))
        self._after_answer(result)
        if speak:
            await self._speak(result.response.text)
        self._settle()

    async def _speak(self, text: str) -> None:
        try:
            _, tts = await self.runtime.voice()
        except Exception as exc:
            self.emit({"type": "error", "message": f"voz indisponível: {exc}"})
            return
        self.sink.interrupted = False
        self.state("speaking")
        speaker = VoiceSession(None, None, tts, self.source, self.sink)  # type: ignore[arg-type]
        await speaker.speak(text)

    def _settle(self) -> None:
        voice_on = self.voice_task is not None and not self.voice_task.done()
        self.state("listening" if voice_on else "idle")

    # -- voice --------------------------------------------------------------------

    async def _voice_loop(self, engine: str | None) -> None:
        self.state("loading")
        try:
            stt, tts = await self.runtime.voice(engine)
        except Exception as exc:
            self.emit({"type": "error", "message": f"voz indisponível: {exc}"})
            self.state("idle")
            return
        from jarvis.voice.factory import vad_config

        voice = self.runtime.settings.voice
        session = VoiceSession(
            self.orchestrator,  # type: ignore[arg-type]
            stt,
            tts,
            self.source,
            self.sink,
            conversation=self.conversation,
            vad_config=vad_config(voice),
            stop_phrases=tuple(voice.stop_phrases),
            on_event=self._on_voice_event,
        )
        self.voice_session = session
        self.emit({"type": "voice", "active": True, "stt": stt.name, "tts": tts.name})
        try:
            await session.run()
        finally:
            self.voice_session = None
            self.emit({"type": "voice", "active": False})
            self.state("idle")

    def _on_voice_event(self, phase: Phase, data: Any) -> None:
        if phase is Phase.LISTENING:
            self.sink.interrupted = False
            self.state("listening")
        elif phase is Phase.HEARD:
            self.emit({"type": "transcript", "text": data.text, "stt_ms": round(data.latency_ms)})
        elif phase is Phase.THINKING:
            self.state("thinking")
        elif phase is Phase.ANSWER:
            self.emit(answer_event(data, "voice"))
            self._after_answer(data)
        elif phase is Phase.SPEAKING:
            self.state("speaking")
        elif phase is Phase.ERROR:
            self.emit({"type": "error", "message": str(data)})

    async def _stop_voice(self) -> None:
        if self.voice_task is not None and not self.voice_task.done():
            self.voice_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.voice_task
        self.voice_task = None
        self.sink.stop()
        self.emit({"type": "stop_audio"})
        self.state("idle")
