"""The voice loop: listen → transcribe → think → speak, turn after turn.

- Half-duplex: JARVIS never transcribes the microphone while it speaks, so it never
  hears (and answers) itself.
- Barge-in (optional): while JARVIS speaks, the microphone is only scored for
  "Hey Jarvis"; hearing it stops the voice at once and JARVIS listens again.
- The conversation object is shared with the text mode, so context carries over
  ("explique Docker" … "e como instalo no Mac?").
- Streaming on the way out: the answer is split into sentences and the next one is
  synthesized while the current one plays, so the first words come out quickly.
- Optional wake word: JARVIS sleeps until "Hey Jarvis", answers turn after turn, and
  goes back to sleep after a silence or "tchau JARVIS". While asleep, audio is only
  scored locally and discarded.
"""

from __future__ import annotations

import asyncio
import dataclasses
import difflib
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.conversation import Conversation
from jarvis.core.errors import AllAgentsFailedError
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.types import OrchestratorResult
from jarvis.voice.audio import AudioClip
from jarvis.voice.devices import AudioSink, AudioSource
from jarvis.voice.stt import STTProvider, Transcript
from jarvis.voice.text import speakable, split_sentences
from jarvis.voice.tts import TTSProvider
from jarvis.voice.vad import EnergyVad, VadConfig, VadEvent
from jarvis.voice.wakeword import WakeWord

DEFAULT_STOP_PHRASES = ("tchau jarvis", "encerrar conversa", "pode parar jarvis", "desligar jarvis")
VOICE_STYLE = (
    "Esta conversa é por voz: sua resposta será falada em voz alta. Responda em no máximo "
    "três frases curtas, sem listas, tabelas, Markdown ou emojis, a menos que o usuário peça "
    "detalhes. Se for preciso mostrar código, diga apenas que o código está na tela."
)
STOP_SIMILARITY = 0.7  # fuzzy match, only for short utterances of 2+ words
# Whisper spells some words in more than one way.
SPELLING_VARIANTS = {"xau": "tchau", "chau": "tchau", "jarbas": "jarvis", "jarvys": "jarvis"}
FAILURE_MESSAGE = "Desculpe, nenhum dos meus agentes gratuitos conseguiu responder agora."
GOODBYE_MESSAGE = "Até logo."
WAKE_REPLY = "Sim?"


class Phase(StrEnum):
    SLEEPING = "sleeping"  # waiting for the wake word
    AWAKE = "awake"  # wake word heard
    LISTENING = "listening"
    HEARD = "heard"
    THINKING = "thinking"
    ANSWER = "answer"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"  # "Hey Jarvis" while speaking
    IDLE = "idle"
    ERROR = "error"


class TurnOutcome(StrEnum):
    ANSWERED = "answered"
    NOTHING_HEARD = "nothing_heard"
    STOP = "stop"
    FAILED = "failed"


@dataclass
class TurnResult:
    outcome: TurnOutcome
    transcript: Transcript | None = None
    result: OrchestratorResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    interrupted: bool = False  # the user cut the answer off


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = "".join(ch if ch.isalnum() else " " for ch in text).split()
    return " ".join(SPELLING_VARIANTS.get(word, word) for word in words)


class VoiceSession:
    def __init__(
        self,
        orchestrator: Orchestrator,
        stt: STTProvider,
        tts: TTSProvider,
        source: AudioSource,
        sink: AudioSink,
        *,
        conversation: Conversation | None = None,
        vad_config: VadConfig | None = None,
        stop_phrases: tuple[str, ...] = DEFAULT_STOP_PHRASES,
        on_event: Callable[[Phase, Any], None] | None = None,
        wake_word: WakeWord | None = None,
        wake_threshold: float = 0.5,
        barge_in: WakeWord | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.stt = stt
        self.tts = tts
        self.source = source
        self.sink = sink
        self.conversation = conversation or Conversation()
        self.vad_config = vad_config or VadConfig()
        self.stop_phrases = tuple(_normalize(p) for p in stop_phrases)
        self.on_event = on_event or (lambda _phase, _data: None)
        self.wake_word = wake_word
        self.wake_threshold = wake_threshold
        self.barge_in = barge_in  # may be the same detector as wake_word: never used at once

    # -- listening ---------------------------------------------------------------

    async def listen(self) -> AudioClip | None:
        """Wait for one utterance. ``None`` when nobody spoke before the timeout."""
        vad = EnergyVad(dataclasses.replace(self.vad_config, sample_rate=self.source.sample_rate))
        self.on_event(Phase.LISTENING, None)
        frames = self.source.frames()
        try:
            async for frame in frames:
                event = vad.feed(frame)
                if event is VadEvent.SPEECH_END:
                    return AudioClip(vad.utterance(), self.source.sample_rate)
                if event is VadEvent.TIMEOUT:
                    return None
        finally:
            await frames.aclose()  # closes the microphone: half-duplex
        return None

    async def wait_for_wake(self) -> float:
        """Sleep until the wake word is heard. Returns the detection score."""
        assert self.wake_word is not None
        self.wake_word.reset()
        self.on_event(Phase.SLEEPING, None)
        frames = self.source.frames()
        try:
            async for frame in frames:
                score = self.wake_word.feed(frame)
                if score >= self.wake_threshold:
                    self.on_event(Phase.AWAKE, score)
                    return score
        finally:
            await frames.aclose()
        return 0.0

    def is_stop(self, text: str) -> bool:
        normalized = _normalize(text)
        if any(phrase in normalized for phrase in self.stop_phrases):
            return True
        # Short utterances that sound like a stop phrase ("jao jarvis" ≈ "tchau jarvis").
        words = len(normalized.split())
        return words >= 2 and any(
            abs(words - len(phrase.split())) <= 1
            and difflib.SequenceMatcher(None, normalized, phrase).ratio() >= STOP_SIMILARITY
            for phrase in self.stop_phrases
        )

    # -- speaking -----------------------------------------------------------------

    async def speak(self, text: str, *, interruptible: bool = True) -> bool:
        """Say ``text``. Returns True when the user cut JARVIS off (barge-in)."""
        sentences = split_sentences(speakable(text))
        if not sentences:
            return False
        self.on_event(Phase.SPEAKING, text)
        playback = asyncio.create_task(self._play(sentences))
        if self.barge_in is None or not interruptible:
            await playback
            return False
        watcher = asyncio.create_task(self._watch_for_barge_in())
        try:
            await asyncio.wait({playback, watcher}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (playback, watcher):
                task.cancel()
            await asyncio.gather(playback, watcher, return_exceptions=True)
        if watcher.cancelled() or watcher.exception() is not None or not watcher.result():
            # JARVIS finished first (or the detector failed): surface TTS errors as before.
            if not playback.cancelled() and playback.exception() is not None:
                raise playback.exception()
            return False
        self.sink.stop()
        self.on_event(Phase.INTERRUPTED, watcher.result())
        return True

    async def _play(self, sentences: list[str]) -> None:
        pending = asyncio.create_task(self.tts.synthesize(sentences[0]))
        try:
            for index in range(len(sentences)):
                clip = await pending
                if index + 1 < len(sentences):
                    pending = asyncio.create_task(self.tts.synthesize(sentences[index + 1]))
                await self.sink.play(clip)
            await self.sink.drained()
        finally:
            if not pending.done():
                pending.cancel()

    async def _watch_for_barge_in(self) -> float:
        """Score the microphone for the wake word while JARVIS talks. Nothing else is kept."""
        assert self.barge_in is not None
        self.barge_in.reset()
        frames = self.source.frames()
        try:
            async for frame in frames:
                score = self.barge_in.feed(frame)
                if score >= self.wake_threshold:
                    return score
        finally:
            await frames.aclose()
        return 0.0

    def interrupt(self) -> None:
        self.sink.stop()

    # -- one turn -----------------------------------------------------------------

    async def turn(self) -> TurnResult:
        timings: dict[str, float] = {}
        clip = await self.listen()
        if clip is None:
            return TurnResult(TurnOutcome.NOTHING_HEARD)

        transcript = await self.stt.transcribe(clip)
        timings["stt_ms"] = transcript.latency_ms
        if transcript.is_empty:
            return TurnResult(TurnOutcome.NOTHING_HEARD, transcript, timings=timings)
        self.on_event(Phase.HEARD, transcript)

        if self.is_stop(transcript.text):
            await self.speak(GOODBYE_MESSAGE)
            return TurnResult(TurnOutcome.STOP, transcript, timings=timings)

        self.on_event(Phase.THINKING, None)
        started = time.perf_counter()
        try:
            result = await self.orchestrator.ask(
                self.conversation, transcript.text, style=VOICE_STYLE
            )
        except AllAgentsFailedError as exc:
            self.on_event(Phase.ERROR, exc)
            await self.speak(FAILURE_MESSAGE)
            return TurnResult(TurnOutcome.FAILED, transcript, timings=timings)
        timings["think_ms"] = (time.perf_counter() - started) * 1000
        self.on_event(Phase.ANSWER, result)

        started = time.perf_counter()
        interrupted = await self.speak(result.response.text)
        timings["speak_ms"] = (time.perf_counter() - started) * 1000
        self.on_event(Phase.IDLE, None)
        return TurnResult(TurnOutcome.ANSWERED, transcript, result, timings, interrupted)

    async def run(self, *, max_turns: int | None = None, stop_on_silence: bool = False) -> int:
        """Continuous conversation. Returns the number of answered turns.

        With a wake word: sleep → "Hey Jarvis" → turns → back to sleep on silence or
        a stop phrase. Without one: turns until a stop phrase.
        """
        if self.wake_word is not None:
            return await self._run_with_wake_word(max_turns)
        answered = turns = 0
        while max_turns is None or turns < max_turns:
            outcome = (await self.turn()).outcome
            if outcome is TurnOutcome.STOP:
                break
            if outcome is TurnOutcome.NOTHING_HEARD:
                if stop_on_silence:
                    break
                continue
            turns += 1
            answered += outcome is TurnOutcome.ANSWERED
        return answered

    async def _run_with_wake_word(self, max_turns: int | None) -> int:
        answered = turns = 0
        while max_turns is None or turns < max_turns:
            await self.wait_for_wake()
            # The end of the same "Hey Jarvis" would cut "Sim?" off: it is short anyway.
            await self.speak(WAKE_REPLY, interruptible=False)
            while max_turns is None or turns < max_turns:
                outcome = (await self.turn()).outcome
                if outcome in (TurnOutcome.STOP, TurnOutcome.NOTHING_HEARD):
                    break  # back to sleep
                turns += 1
                answered += outcome is TurnOutcome.ANSWERED
        return answered
