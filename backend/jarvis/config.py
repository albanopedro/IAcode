"""Configuration: ``config/agents.toml`` (no secrets) + ``.env`` (secrets only).

COST_MODE comes from the environment and defaults to FREE_ONLY. An invalid
value also falls back to FREE_ONLY — a typo must never unlock paid agents.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.core.types import Capability, CostClass, CostMode, Privacy, TaskType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "agents.toml"
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class OpenCodeModelConfig(BaseModel):
    id: str
    name: str | None = None
    priority: int = 50
    privacy: Privacy = Privacy.UNKNOWN
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    quality: dict[TaskType, float] = Field(default_factory=dict)
    rpm_limit: int | None = None


class OpenCodeConfig(BaseModel):
    enabled: bool = True
    binary: str = "opencode"
    timeout: float = 120.0
    models: list[OpenCodeModelConfig] = Field(default_factory=list)


class OpenAICompatConfig(BaseModel):
    id: str
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    enabled: bool = True
    cost_class: CostClass = CostClass.FREE_WITH_LIMITS
    free_model_pattern: str | None = None
    health_check: str = "models"
    daily_limit: int | None = None
    rpm_limit: int | None = None
    quota_group: str | None = None
    context_window: int | None = None
    priority: int = 50
    privacy: Privacy = Privacy.UNKNOWN
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    quality: dict[TaskType, float] = Field(default_factory=dict)
    allow_paid: bool = False


class VoiceConfig(BaseModel):
    """Local, free voice. Models are downloaded once into ``<data_dir>/models``."""

    language: str = "pt"
    stt_model: str = "large-v3-turbo-q5_0"
    stt_threads: int = 4
    # A natural sentence works better than a word list (tested: "Docker" was heard
    # as "do Querer" without it).
    stt_prompt: str = "Olá, JARVIS. Falamos de Docker, Python, GitHub, Linux, macOS e programação."
    tts_engine: str = "say"  # "say" (macOS, built in) or "piper" (open source)
    say_voice: str = "Luciana"
    say_rate: int | None = None
    piper_voice: str = "pt_BR-faber-medium"
    silence_ms: int = 900
    no_speech_timeout: float = 8.0
    max_utterance_seconds: float = 30.0
    stop_phrases: list[str] = Field(
        default_factory=lambda: [
            "tchau jarvis",
            "encerrar conversa",
            "pode parar jarvis",
            "desligar jarvis",
        ]
    )


class Settings(BaseModel):
    cost_mode: CostMode = CostMode.FREE_ONLY
    data_dir: Path = DEFAULT_DATA_DIR
    health_interval: float = 120.0
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    opencode: OpenCodeConfig = Field(default_factory=OpenCodeConfig)
    openai_compat: list[OpenAICompatConfig] = Field(default_factory=list)


def load_dotenv(path: Path = DEFAULT_ENV) -> None:
    """Minimal .env reader. Real environment variables always win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def cost_mode_from_env() -> CostMode:
    raw = os.environ.get("COST_MODE", "").strip().upper()
    try:
        return CostMode(raw) if raw else CostMode.FREE_ONLY
    except ValueError:
        return CostMode.FREE_ONLY


def load_settings(config_path: Path | None = None, env_path: Path | None = None) -> Settings:
    load_dotenv(env_path or DEFAULT_ENV)
    path = config_path or Path(os.environ.get("JARVIS_CONFIG", DEFAULT_CONFIG))
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data.pop("cost_mode", None)  # the file cannot change the cost mode
    settings = Settings.model_validate(data)
    settings.cost_mode = cost_mode_from_env()
    if os.environ.get("JARVIS_DATA_DIR"):
        settings.data_dir = Path(os.environ["JARVIS_DATA_DIR"])
    return settings
