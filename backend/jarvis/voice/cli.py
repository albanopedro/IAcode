"""Voice commands for the CLI (imported only when a voice command runs).

jarvis voice [--tts say|piper] [--once]   # talk to JARVIS (microphone + speaker)
jarvis speak "texto" [--tts piper]        # test the voice
jarvis transcribe arquivo.wav             # test speech-to-text on a file
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from jarvis.config import load_settings
from jarvis.core.health_monitor import HealthMonitor
from jarvis.core.types import OrchestratorResult
from jarvis.voice.audio import read_wav
from jarvis.voice.devices import Microphone, Speaker
from jarvis.voice.factory import build_stt, build_tts, vad_config
from jarvis.voice.session import Phase, VoiceSession
from jarvis.voice.stt import Transcript

LABELS = {
    Phase.SLEEPING: "💤 dormindo… diga “Hey Jarvis” (pronúncia em inglês) para acordar",
    Phase.AWAKE: "⚡ acordei!",
    Phase.LISTENING: "🎙️  ouvindo… (fale quando quiser; diga “tchau JARVIS” para sair)",
    Phase.THINKING: "💭 pensando…",
    Phase.SPEAKING: "🔊 falando…",
    Phase.INTERRUPTED: "✋ interrompido, pode falar",
}


def print_event(phase: Phase, data: Any) -> None:
    if phase in LABELS:
        print(LABELS[phase], flush=True)
    elif phase is Phase.HEARD and isinstance(data, Transcript):
        print(f"Você: {data.text}   ({data.latency_ms / 1000:.1f}s para transcrever)")
    elif phase is Phase.ANSWER and isinstance(data, OrchestratorResult):
        agent = data.attempts[-1].agent_id if data.attempts else "?"
        print(f"JARVIS: {data.response.text}\n   [{data.task.value}] {agent}")
    elif phase is Phase.ERROR:
        print(f"⚠️  {data}")


async def cmd_voice(tts_engine: str | None, once: bool, wake: bool = False) -> int:
    from jarvis.cli import build, open_conversation, shutdown

    settings = load_settings()
    manager, assistant, interval = build()
    stt, tts = build_stt(settings), build_tts(settings, tts_engine)
    print(f"Carregando voz: {stt.name} + {tts.name}…", flush=True)
    started = time.perf_counter()
    await asyncio.gather(asyncio.to_thread(stt.warm_up), asyncio.to_thread(tts.warm_up))
    print(f"Pronto em {time.perf_counter() - started:.1f}s.\n")

    wake_word = barge_in = None
    if wake or settings.voice.barge_in:
        from jarvis.voice.factory import build_wake_word

        detector = build_wake_word(settings)
        try:
            await asyncio.to_thread(detector.warm_up)
        except Exception as exc:
            if wake:
                raise
            print(f"(sem interrupção por voz: {exc})")
        else:
            wake_word = detector if wake else None
            barge_in = detector if settings.voice.barge_in else None
    if barge_in is not None:
        print("Diga “Hey Jarvis” enquanto eu falo para me interromper.")
    session = VoiceSession(
        assistant,  # type: ignore[arg-type]  # same ask() as the orchestrator, plus memory
        stt,
        tts,
        Microphone(),
        Speaker(),
        conversation=open_conversation(assistant),
        vad_config=vad_config(settings.voice),
        stop_phrases=tuple(settings.voice.stop_phrases),
        wake_word=wake_word,
        wake_threshold=settings.voice.wake_threshold,
        barge_in=barge_in,
        on_event=print_event,
    )
    monitor = HealthMonitor(manager, interval)
    monitor.start()
    try:
        await session.run(max_turns=1 if once else None)
    except KeyboardInterrupt:
        pass
    finally:
        session.interrupt()
        await monitor.stop()
        await shutdown(manager, assistant)
    return 0


async def cmd_speak(text: str, tts_engine: str | None) -> int:
    settings = load_settings()
    tts = build_tts(settings, tts_engine)
    session = VoiceSession(None, None, tts, None, Speaker())  # type: ignore[arg-type]
    await asyncio.to_thread(tts.warm_up)
    await session.speak(text)
    return 0


async def cmd_transcribe(path: str) -> int:
    settings = load_settings()
    stt = build_stt(settings)
    transcript = await stt.transcribe(read_wav(path))
    print(transcript.text)
    print(f"({transcript.audio_seconds:.1f}s de áudio em {transcript.latency_ms / 1000:.1f}s)")
    return 0
