"""Audio in and out.

- ``Microphone`` / ``Speaker`` use sounddevice (PortAudio, bundled in the wheel).
  The first use asks macOS for microphone permission for the terminal app.
- ``ClipSource`` / ``MemorySink`` are drop-in replacements that read from and
  write to memory: they make the whole voice loop testable without hardware.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from jarvis.voice.audio import STT_SAMPLE_RATE, AudioClip
from jarvis.voice.vad import FRAME_MS


class AudioSource(Protocol):
    sample_rate: int

    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield consecutive 30 ms float32 frames. Closing the iterator releases the device."""
        ...


class AudioSink(Protocol):
    async def play(self, clip: AudioClip) -> None: ...

    def stop(self) -> None: ...


class Microphone:
    def __init__(self, sample_rate: int = STT_SAMPLE_RATE, device: int | str | None = None):
        self.sample_rate = sample_rate
        self.device = device

    async def frames(self) -> AsyncIterator[np.ndarray]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=400)
        frame = self.sample_rate * FRAME_MS // 1000

        def callback(indata, _frames, _time, _status) -> None:
            data = indata[:, 0].copy()
            with contextlib.suppress(RuntimeError):  # loop closed while stopping
                loop.call_soon_threadsafe(_put_nowait, queue, data)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame,
            device=self.device,
            callback=callback,
        )
        with stream:
            while True:
                yield await queue.get()


def _put_nowait(queue: asyncio.Queue, item) -> None:
    with contextlib.suppress(asyncio.QueueFull):  # drop audio rather than block PortAudio
        queue.put_nowait(item)


class Speaker:
    def __init__(self, device: int | str | None = None) -> None:
        self.device = device

    async def play(self, clip: AudioClip) -> None:
        import sounddevice as sd

        if clip.duration == 0:
            return
        sd.play(clip.samples, clip.sample_rate, device=self.device)
        await asyncio.to_thread(sd.wait)

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()


class ClipSource:
    """Feeds clips into the pipeline as if they were spoken into a microphone.

    Each clip is preceded by a short silence (so the VAD can calibrate) and followed
    by enough silence to end the utterance. The position is kept across ``listen``
    calls, like a real microphone; once everything was played, it yields silence.
    """

    def __init__(self, clips: list[AudioClip], *, trailing_silence: float = 1.5):
        self.sample_rate = STT_SAMPLE_RATE
        self.frame = self.sample_rate * FRAME_MS // 1000
        self._silence = np.zeros(self.frame, dtype=np.float32)
        self._queue: deque[np.ndarray] = deque()
        lead = int(600 / FRAME_MS)
        tail = int(trailing_silence * 1000 / FRAME_MS)
        for clip in clips:
            samples = clip.resampled(self.sample_rate).samples
            self._queue.extend([self._silence] * lead)
            for start in range(0, len(samples), self.frame):
                chunk = samples[start : start + self.frame]
                if len(chunk) < self.frame:
                    chunk = np.pad(chunk, (0, self.frame - len(chunk)))
                self._queue.append(chunk.astype(np.float32))
            self._queue.extend([self._silence] * tail)

    @property
    def exhausted(self) -> bool:
        return not self._queue

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            yield self._queue.popleft() if self._queue else self._silence
            await asyncio.sleep(0)


class MemorySink:
    def __init__(self) -> None:
        self.played: list[AudioClip] = []
        self.stopped = 0

    async def play(self, clip: AudioClip) -> None:
        self.played.append(clip)

    def stop(self) -> None:
        self.stopped += 1
