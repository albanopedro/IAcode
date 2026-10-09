"""One browser connection: text chat, continuous voice and live events.

Client → server (JSON text messages, plus binary microphone audio):
    {"type": "text", "text": "...", "speak": false}   ask in writing (optionally hear it)
    {"type": "voice_start", "tts": "say" | "piper", "wake": true}   start the voice loop
                                                       (wake: sleep until "Hey Jarvis")
    {"type": "voice_stop"}                             stop listening
    {"type": "interrupt"}                              stop JARVIS talking
    {"type": "played", "count": 3}                     sentences the browser finished playing
    {"type": "clear"} / {"type": "new_conversation"}   start a new conversation
    {"type": "open_conversation", "id": "..."}         reopen a saved conversation
    {"type": "delete_conversation", "id": "..."}       delete a saved conversation
    {"type": "list_conversations"}                     saved conversations
    {"type": "list_facts"} / {"type": "forget_fact", "id": 1}
    {"type": "clear_facts", "confirm": true}           forget every long-term fact
    {"type": "confirm_reply", "id": "...", "approved": true}   answer a tool confirmation
    {"type": "set_local_only", "value": true}          only local models answer (nothing leaves)
    {"type": "status"}                                 ask for the agents' status
    <binary> PCM16 mono 16 kHz frames while the voice loop runs

Server → client (JSON), plus binary WAV messages with JARVIS's voice:
    hello · state · transcript · answer · agents · stop_audio · cleared · error
    voice (the loop started/stopped) · wake ("Hey Jarvis" heard) · interrupted (barge-in)
    history (the open conversation) · conversations · facts
    tool (a tool started/finished) · confirm / confirm_closed (a tool needs your OK)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from jarvis.core.conversation import Conversation
from jarvis.core.errors import AllAgentsFailedError
from jarvis.core.types import OrchestratorResult
from jarvis.server.runtime import JarvisRuntime
from jarvis.server.ws_audio import QueueAudioSource, WebSocketSink
from jarvis.voice.session import Phase, VoiceSession

MAX_TEXT = 4000
CONFIRM_TIMEOUT = 60  # seconds to answer a confirmation in the browser
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
        "tools": [t.model_dump() for t in result.tools],
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
                "private": turn.private,
            }
            for turn in conversation.turns
        ],
    }


class _LockedOrchestrator:
    """Text and voice share one conversation: never answer two messages at once.

    Also gives every request a tool context whose confirmations go to the browser.
    """

    def __init__(self, assistant, lock: asyncio.Lock, tool_context, local_only) -> None:
        self._assistant = assistant
        self._lock = lock
        self._tool_context = tool_context
        self._local_only = local_only

    async def ask(self, conversation, text, *, style=None):
        async with self._lock:
            return await self._assistant.ask(
                conversation,
                text,
                style=style,
                tool_context=self._tool_context(),
                local_only=self._local_only(),
            )


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
        self.local_only = False  # "só local": only models running on this Mac answer
        self.orchestrator = _LockedOrchestrator(
            runtime.assistant, asyncio.Lock(), self._tool_context, lambda: self.local_only
        )
        self.pending_confirms: dict[str, asyncio.Future] = {}
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
                "local_only": self.local_only,
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
                wake = message.get("wake") is True
                self.voice_task = asyncio.create_task(self._voice_loop(engine, wake))
        elif kind == "voice_stop":
            await self._stop_voice()
        elif kind == "interrupt":
            self.sink.stop()
            self.emit({"type": "stop_audio"})
        elif kind == "played":
            count = message.get("count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                self.sink.mark_played(count)
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
        elif kind == "set_local_only":
            self.local_only = message.get("value") is True
            self.emit({"type": "local_only", "value": self.local_only})
        elif kind == "confirm_reply":
            future = self.pending_confirms.pop(str(message.get("id")), None)
            if future is not None and not future.done():
                future.set_result(message.get("approved") is True)
        elif kind == "status":
            self.emit(agents_event(self.runtime))
        else:
            self.emit({"type": "error", "message": f"tipo desconhecido: {kind}"})

    # -- tools --------------------------------------------------------------------

    def _tool_context(self):
        from jarvis.tools.base import ToolContext

        def on_event(kind: str, data: dict) -> None:
            if kind in ("tool_start", "tool_end"):
                self.emit({"type": "tool", "phase": kind.removeprefix("tool_"), **data})

        return ToolContext(confirm=self._confirm, on_event=on_event)

    async def _confirm(self, call, tool, reason: str) -> bool:
        """Ask the browser; no answer within the time limit means NO."""
        confirm_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending_confirms[confirm_id] = future
        self.emit(
            {
                "type": "confirm",
                "id": confirm_id,
                "tool": tool.name,
                "title": tool.title,
                "risk": tool.risk.name.lower(),
                "description": tool.describe_call(call.args),
                "reason": reason,
                "timeout": CONFIRM_TIMEOUT,
            }
        )
        try:
            return await asyncio.wait_for(future, CONFIRM_TIMEOUT)
        except TimeoutError:
            return False
        finally:
            self.pending_confirms.pop(confirm_id, None)
            self.emit({"type": "confirm_closed", "id": confirm_id})

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

    async def _voice_loop(self, engine: str | None, wake: bool = False) -> None:
        self.state("loading")
        wake_word = None
        try:
            stt, tts = await self.runtime.voice(engine)
            if wake:
                wake_word = await self.runtime.wake_word()
        except Exception as exc:
            self.emit({"type": "error", "message": f"voz indisponível: {exc}"})
            self.state("idle")
            return
        barge_in = wake_word
        if barge_in is None and self.runtime.settings.voice.barge_in:
            with contextlib.suppress(Exception):  # optional: without it, the button still works
                barge_in = await self.runtime.wake_word()
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
            wake_word=wake_word,
            wake_threshold=voice.wake_threshold,
            barge_in=barge_in if voice.barge_in else None,
        )
        self.voice_session = session
        self.emit(
            {
                "type": "voice",
                "active": True,
                "stt": stt.name,
                "tts": tts.name,
                "wake": wake,
                "barge_in": session.barge_in is not None,
            }
        )
        try:
            await session.run()
        finally:
            self.voice_session = None
            self.emit({"type": "voice", "active": False})
            self.state("idle")

    def _on_voice_event(self, phase: Phase, data: Any) -> None:
        if phase is Phase.SLEEPING:
            self.state("sleeping")
        elif phase is Phase.AWAKE:
            self.emit({"type": "wake", "score": round(float(data), 3)})
        elif phase is Phase.LISTENING:
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
        elif phase is Phase.INTERRUPTED:
            self.emit({"type": "stop_audio"})
            self.emit({"type": "interrupted", "score": round(float(data), 3)})
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
