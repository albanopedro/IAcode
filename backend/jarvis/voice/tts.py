"""Text-to-speech providers. Both are local and free."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from jarvis.voice.audio import AudioClip, read_wav


class TTSError(RuntimeError):
    pass


class TTSProvider(ABC):
    name: str

    @abstractmethod
    async def synthesize(self, text: str) -> AudioClip: ...

    def warm_up(self) -> None:  # noqa: B027 — optional hook
        """Load models ahead of the first sentence."""


class MacSayTTS(TTSProvider):
    """The macOS built-in ``say`` command. Zero setup; not open source."""

    def __init__(self, voice: str = "Luciana", rate: int | None = None) -> None:
        self.voice = voice
        self.rate = rate
        self.name = f"macOS say · {voice}"

    async def synthesize(self, text: str) -> AudioClip:
        binary = shutil.which("say")
        if binary is None:
            raise TTSError("the macOS 'say' command is not available")
        with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as tmp:
            out = Path(tmp) / "speech.wav"
            args = [binary, "-v", self.voice, "--file-format=WAVE", "--data-format=LEI16@22050"]
            if self.rate:
                args += ["-r", str(self.rate)]
            # "--" so a sentence starting with "-" is never read as an option.
            proc = await asyncio.create_subprocess_exec(
                *args,
                "-o",
                str(out),
                "--",
                text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not out.exists():
                raise TTSError(f"say failed: {stderr.decode(errors='replace').strip()}")
            return read_wav(out)


class PiperTTS(TTSProvider):
    """Piper (open source, GPL-3.0) with a downloaded ONNX voice, e.g. pt_BR-faber-medium.

    The voice (~60 MB) is downloaded once from the official rhasspy/piper-voices
    repository on Hugging Face into ``voices_dir``.
    """

    def __init__(self, voice: str = "pt_BR-faber-medium", voices_dir: str | Path = "voices"):
        self.voice_name = voice
        self.voices_dir = Path(voices_dir)
        self.name = f"Piper · {voice}"
        self._voice = None

    def model_path(self) -> Path:
        return self.voices_dir / f"{self.voice_name}.onnx"

    def warm_up(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice

        if not self.model_path().exists():
            from piper.download_voices import download_voice

            self.voices_dir.mkdir(parents=True, exist_ok=True)
            download_voice(self.voice_name, self.voices_dir)
        self._voice = PiperVoice.load(self.model_path())

    def _run(self, text: str) -> AudioClip:
        self.warm_up()
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return AudioClip(np.zeros(0, dtype=np.float32), self._voice.config.sample_rate)
        ints = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        return AudioClip.from_int16(ints, chunks[0].sample_rate)

    async def synthesize(self, text: str) -> AudioClip:
        return await asyncio.to_thread(self._run, text)
