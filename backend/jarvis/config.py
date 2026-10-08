"""Configuration: ``config/agents.toml`` (no secrets) + ``.env`` (secrets only).

COST_MODE comes from the environment and defaults to FREE_ONLY. An invalid
value also falls back to FREE_ONLY — a typo must never unlock paid agents.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.core.types import Capability, CostClass, CostMode, Privacy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "agents.toml"
DEFAULT_ENV = PROJECT_ROOT / ".env"


class OpenCodeModelConfig(BaseModel):
    id: str
    name: str | None = None
    priority: int = 50
    privacy: Privacy = Privacy.UNKNOWN
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})


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
    daily_limit: int | None = None
    priority: int = 50
    privacy: Privacy = Privacy.UNKNOWN
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    allow_paid: bool = False


class Settings(BaseModel):
    cost_mode: CostMode = CostMode.FREE_ONLY
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
    return settings
