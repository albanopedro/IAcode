"""Audio over the WebSocket: the browser is the microphone and the speaker.

Microphone frames arrive as binary messages of 16-bit PCM, mono, 16 kHz.
JARVIS's voice goes back as WAV bytes, one message per sentence.

Half-duplex: while the browser plays JARVIS's voice it stops sending microphone
audio, and every new ``listen`` drops frames left over from before it started.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from jarvis.voice.audio import STT_SAMPLE_RATE, AudioClip, write_wav

MAX_FRAME_BYTES = 64 * 1024
QUEUE_FRAMES = 1000  # ~30 s of audio at 30 ms frames


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

    def __init__(self, send_audio: Callable[[bytes, float], Awaitable[None]]) -> None:
        self._send_audio = send_audio
        self.interrupted = False

    async def play(self, clip: AudioClip) -> None:
        if self.interrupted or clip.duration == 0:
            return
        await self._send_audio(wav_bytes(clip), clip.duration)

    def stop(self) -> None:
        self.interrupted = True
