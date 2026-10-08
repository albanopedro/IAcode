"""One browser connection: text chat, continuous voice and live events.

Client → server (JSON text messages, plus binary microphone audio):
    {"type": "text", "text": "...", "speak": false}   ask in writing (optionally hear it)
    {"type": "voice_start", "tts": "say" | "piper"}    start the continuous voice loop
    {"type": "voice_stop"}                             stop listening
    {"type": "interrupt"}                              stop JARVIS talking
    {"type": "clear"}                                  forget the conversation
    {"type": "status"}                                 ask for the agents' status
    <binary> PCM16 mono 16 kHz frames while the voice loop runs

Server → client (JSON), plus binary WAV messages with JARVIS's voice:
    hello · state · transcript · answer · agents · stop_audio · cleared · error
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
        self.conversation = Conversation()
        self.source = QueueAudioSource()
        self.sink = WebSocketSink(self._send_audio)
        self.orchestrator = _LockedOrchestrator(runtime.orchestrator, asyncio.Lock())
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
        elif kind == "clear":
            self.conversation.clear()
            self.emit({"type": "cleared"})
        elif kind == "status":
            self.emit(agents_event(self.runtime))
        else:
            self.emit({"type": "error", "message": f"tipo desconhecido: {kind}"})

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
        self.emit(agents_event(self.runtime))
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
        self.emit({"type": "voice", "active": True, "stt": stt.name, "tts": tts.name})
        try:
            await session.run()
        finally:
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
            self.emit(agents_event(self.runtime))
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
