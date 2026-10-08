"""Speech-to-text providers."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jarvis.voice.audio import STT_SAMPLE_RATE, AudioClip

# Whisper invents these on silence or noise; never send them to the assistant.
HALLUCINATIONS = frozenset(
    {
        "",
        "legendas pela comunidade amara.org",
        "obrigado.",
        "obrigado por assistir.",
        "tchau.",
        "[música]",
        "[silêncio]",
        "(música)",
        "...",
    }
)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    audio_seconds: float
    latency_ms: float

    @property
    def is_empty(self) -> bool:
        return self.text.strip().lower() in HALLUCINATIONS


class STTProvider(ABC):
    name: str

    @abstractmethod
    async def transcribe(self, clip: AudioClip) -> Transcript: ...

    def warm_up(self) -> None:  # noqa: B027 — optional hook
        """Load models ahead of the first request."""


class WhisperCppSTT(STTProvider):
    """whisper.cpp (Metal on Apple Silicon) through pywhispercpp. Fully local.

    The model file is downloaded once from the official whisper.cpp repository on
    Hugging Face into ``models_dir`` and reused afterwards (no account, no cost).
    ``large-v3-turbo-q5_0`` (~550 MB) is accurate in Portuguese and fast on an M4.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo-q5_0",
        *,
        models_dir: str | Path | None = None,
        language: str | None = "pt",
        threads: int = 4,
        prompt: str = "",
    ) -> None:
        self.model_name = model
        self.prompt = prompt
        self.models_dir = Path(models_dir) if models_dir else None
        self.language = language
        self.threads = threads
        self.name = f"whisper.cpp · {model}"
        self._model = None

    def warm_up(self) -> None:
        if self._model is None:
            from pywhispercpp.model import Model

            if self.models_dir is not None:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            self._model = Model(
                self.model_name,
                models_dir=str(self.models_dir) if self.models_dir else None,
                n_threads=self.threads,
                print_progress=False,
                print_realtime=False,
                redirect_whispercpp_logs_to=None,
            )

    def _run(self, samples: np.ndarray) -> str:
        self.warm_up()
        params = {"language": self.language or "auto", "translate": False, "no_context": True}
        if self.prompt:
            # Vocabulary hint: helps Whisper spell names and technical terms.
            params["initial_prompt"] = self.prompt
        segments = self._model.transcribe(samples, **params)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, clip: AudioClip) -> Transcript:
        started = time.perf_counter()
        samples = clip.resampled(STT_SAMPLE_RATE).samples
        text = await asyncio.to_thread(self._run, samples) if len(samples) else ""
        return Transcript(
            text=text,
            language=self.language,
            audio_seconds=clip.duration,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
