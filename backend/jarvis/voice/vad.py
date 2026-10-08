"""Voice activity detection: when does the user start and stop talking?

An energy-based detector, deliberately simple and dependency-free:
1. it learns the background noise level from the first frames;
2. speech starts when the energy stays above ``noise × start_ratio`` for a few frames;
3. speech ends after ``silence_ms`` of quiet (or at ``max_seconds``).

It works well for a quiet room and push-to-talk style turns. A neural VAD
(Silero, ONNX) can replace it later behind the same ``Utterance`` contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

FRAME_MS = 30


class VadEvent(StrEnum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TIMEOUT = "timeout"  # no speech at all before the deadline


@dataclass
class VadConfig:
    sample_rate: int = 16_000
    calibration_ms: int = 300
    min_threshold: float = 0.008  # RMS floor, so dead silence never triggers
    start_ratio: float = 3.0
    start_frames: int = 3  # ~90 ms of sound to start
    silence_ms: int = 900
    pre_roll_ms: int = 300  # audio kept from just before the start
    max_seconds: float = 30.0
    no_speech_timeout: float = 8.0


@dataclass
class EnergyVad:
    config: VadConfig = field(default_factory=VadConfig)

    def __post_init__(self) -> None:
        self.reset()

    @property
    def frame_size(self) -> int:
        return self.config.sample_rate * FRAME_MS // 1000

    def reset(self) -> None:
        cfg = self.config
        self.noise_levels: list[float] = []
        self.loud_run = 0
        self.quiet_ms = 0
        self.in_speech = False
        self.elapsed_ms = 0
        self.speech: list[np.ndarray] = []
        self.pre_roll: deque[np.ndarray] = deque(maxlen=max(cfg.pre_roll_ms // FRAME_MS, 1))

    @property
    def noise_floor(self) -> float:
        if not self.noise_levels:
            return self.config.min_threshold
        return float(np.median(self.noise_levels))

    @property
    def threshold(self) -> float:
        return max(self.noise_floor * self.config.start_ratio, self.config.min_threshold)

    def feed(self, frame: np.ndarray) -> VadEvent:
        """Process one 30 ms frame of float32 samples."""
        cfg = self.config
        self.elapsed_ms += FRAME_MS
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0

        if not self.in_speech:
            if self.elapsed_ms <= cfg.calibration_ms:
                self.noise_levels.append(rms)
                self.pre_roll.append(frame)
                return VadEvent.NONE
            self.pre_roll.append(frame)
            self.loud_run = self.loud_run + 1 if rms > self.threshold else 0
            if self.loud_run >= cfg.start_frames:
                self.in_speech = True
                self.quiet_ms = 0
                self.speech = list(self.pre_roll)
                return VadEvent.SPEECH_START
            if self.elapsed_ms >= cfg.no_speech_timeout * 1000:
                return VadEvent.TIMEOUT
            if rms <= self.threshold:
                # Keep adapting to slow changes in background noise.
                self.noise_levels = (self.noise_levels + [rms])[-100:]
            return VadEvent.NONE

        self.speech.append(frame)
        # Ending needs only half the start threshold: trailing syllables are quiet.
        self.quiet_ms = 0 if rms > self.threshold / 2 else self.quiet_ms + FRAME_MS
        speech_ms = len(self.speech) * FRAME_MS
        if self.quiet_ms >= cfg.silence_ms or speech_ms >= cfg.max_seconds * 1000:
            self.in_speech = False
            return VadEvent.SPEECH_END
        return VadEvent.NONE

    def utterance(self) -> np.ndarray:
        if not self.speech:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.speech).astype(np.float32)
