"""Audio clips and WAV helpers (standard library ``wave`` + numpy)."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STT_SAMPLE_RATE = 16_000  # what Whisper expects


@dataclass(frozen=True)
class AudioClip:
    """Mono float32 samples in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @classmethod
    def from_int16(cls, data: bytes | np.ndarray, sample_rate: int) -> AudioClip:
        ints = np.frombuffer(data, dtype=np.int16) if isinstance(data, bytes) else data
        return cls(ints.astype(np.float32) / 32768.0, sample_rate)

    def to_int16(self) -> np.ndarray:
        return (np.clip(self.samples, -1.0, 1.0) * 32767).astype(np.int16)

    def resampled(self, sample_rate: int) -> AudioClip:
        """Linear-interpolation resampling: good enough for speech recognition."""
        if sample_rate == self.sample_rate or len(self.samples) == 0:
            return AudioClip(self.samples.astype(np.float32), sample_rate)
        count = round(len(self.samples) * sample_rate / self.sample_rate)
        positions = np.linspace(0, len(self.samples) - 1, count)
        samples = np.interp(positions, np.arange(len(self.samples)), self.samples)
        return AudioClip(samples.astype(np.float32), sample_rate)


def read_wav(source: str | Path | bytes) -> AudioClip:
    handle = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")  # noqa: SIM115
    with handle, wave.open(handle) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        frames = wav.readframes(wav.getnframes())
        channels = wav.getnchannels()
        rate = wav.getframerate()
    ints = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        ints = ints.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return AudioClip.from_int16(ints, rate)


def write_wav(clip: AudioClip, target: str | Path | io.BytesIO) -> None:
    with wave.open(target if isinstance(target, io.BytesIO) else str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(clip.sample_rate)
        wav.writeframes(clip.to_int16().tobytes())


def concat(clips: list[AudioClip]) -> AudioClip:
    if not clips:
        return AudioClip(np.zeros(0, dtype=np.float32), STT_SAMPLE_RATE)
    rate = clips[0].sample_rate
    return AudioClip(np.concatenate([c.resampled(rate).samples for c in clips]), rate)
