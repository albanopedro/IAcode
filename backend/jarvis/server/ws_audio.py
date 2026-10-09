"""Audio over the WebSocket: the browser is the microphone and the speaker.

Microphone frames arrive as binary messages of 16-bit PCM, mono, 16 kHz.
JARVIS's voice goes back as WAV bytes, one message per sentence.

Half-duplex: every new ``listen`` drops frames left over from before it started. While
the browser plays JARVIS's voice it only keeps sending microphone audio when barge-in is
on, and then those frames are only scored for "Hey Jarvis", never transcribed.

The browser reports how many sentences it has finished playing ({"type": "played"}), so
the server knows when JARVIS has really stopped talking.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from jarvis.voice.audio import STT_SAMPLE_RATE, AudioClip, write_wav

MAX_FRAME_BYTES = 64 * 1024
QUEUE_FRAMES = 1000  # ~30 s of audio at 30 ms frames
DRAIN_SLACK = 2.0  # seconds to wait past the expected end before giving up on the browser


class QueueAudioSource:
    sample_rate = STT_SAMPLE_RATE

    def __init__(self) -> None:
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=QUEUE_FRAMES)

    def push(self, data: bytes) -> bool:
        """Add PCM16 bytes from the browser. Returns False when the frame was rejected."""
        if not data or len(data) > MAX_FRAME_BYTES or len(data) % 2:
            return False
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._queue.put_nowait(samples)
        except asyncio.QueueFull:
            return False
        return True

    def clear(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        self.clear()  # audio recorded while JARVIS was talking is never used
        while True:
            yield await self._queue.get()


def wav_bytes(clip: AudioClip) -> bytes:
    buffer = io.BytesIO()
    write_wav(clip, buffer)
    return buffer.getvalue()


class WebSocketSink:
    """Sends each synthesized sentence to the browser, which plays them in order."""

    def __init__(
        self,
        send_audio: Callable[[bytes, float], Awaitable[None]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send_audio = send_audio
        self._clock = clock
        self.interrupted = False
        self.sent = 0  # sentences sent on this connection
        self.played = 0  # sentences the browser says it finished
        self._busy_until = 0.0  # when the browser should be done, if it never reports
        self._changed = asyncio.Event()

    async def play(self, clip: AudioClip) -> None:
        if self.interrupted or clip.duration == 0:
            return
        await self._send_audio(wav_bytes(clip), clip.duration)
        self.sent += 1
        self._busy_until = max(self._clock(), self._busy_until) + clip.duration

    def mark_played(self, count: int) -> None:
        self.played = max(self.played, min(count, self.sent))
        self._changed.set()

    async def drained(self) -> None:
        while not self.interrupted and self.played < self.sent:
            remaining = self._busy_until - self._clock() + DRAIN_SLACK
            if remaining <= 0:
                return  # the browser never answered: assume it is done
            self._changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), remaining)

    def stop(self) -> None:
        self.interrupted = True
        self._changed.set()
