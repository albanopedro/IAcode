"""Wake word: JARVIS sleeps until it hears "Hey Jarvis".

openWakeWord (code Apache-2.0) runs three small ONNX models locally, 80 ms at a time.
While asleep, audio is only scored on the fly and thrown away: nothing is recorded,
transcribed or sent anywhere until the wake word is detected.

The pretrained "hey_jarvis" model is licensed CC BY-NC-SA 4.0: fine for personal use,
not for a commercial product. It was trained on English pronunciation, so say
"Hey Jarvis" the English way ("Ei Jarvis" with a Brazilian accent scores ~0.26).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

CHUNK = 1280  # 80 ms at 16 kHz: what openWakeWord expects
FEATURE_FILES = ("melspectrogram.onnx", "embedding_model.onnx")


class WakeWord(Protocol):
    name: str

    def feed(self, frame: np.ndarray) -> float:
        """Score float32 16 kHz audio; returns the best score seen in this frame."""
        ...

    def reset(self) -> None: ...


def model_files(models_dir: Path, model: str) -> list[Path]:
    return [models_dir / f"{model}.onnx", *(models_dir / f for f in FEATURE_FILES)]


def is_downloaded(models_dir: Path, model: str) -> bool:
    return all(path.is_file() for path in model_files(models_dir, model))


def download(models_dir: Path, model: str) -> None:
    """One-time download from the official openWakeWord GitHub releases (~9 MB)."""
    from openwakeword import utils

    models_dir.mkdir(parents=True, exist_ok=True)
    utils.download_models([model], target_directory=str(models_dir))


class OpenWakeWord:
    def __init__(self, models_dir: Path, model: str = "hey_jarvis_v0.1", threshold: float = 0.5):
        self.models_dir = Path(models_dir)
        self.model_name = model
        self.threshold = threshold
        self.name = f"openWakeWord · {model}"
        self._model = None
        self._buffer = np.zeros(0, dtype=np.int16)

    def warm_up(self) -> None:
        if self._model is not None:
            return
        if not is_downloaded(self.models_dir, self.model_name):
            download(self.models_dir, self.model_name)
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[str(self.models_dir / f"{self.model_name}.onnx")],
            inference_framework="onnx",
            melspec_model_path=str(self.models_dir / "melspectrogram.onnx"),
            embedding_model_path=str(self.models_dir / "embedding_model.onnx"),
        )

    def feed(self, frame: np.ndarray) -> float:
        self.warm_up()
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        self._buffer = np.concatenate([self._buffer, pcm])
        best = 0.0
        while len(self._buffer) >= CHUNK:
            chunk, self._buffer = self._buffer[:CHUNK], self._buffer[CHUNK:]
            scores = self._model.predict(chunk)
            best = max(best, *scores.values())
        return best

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)
        if self._model is not None:
            self._model.reset()
