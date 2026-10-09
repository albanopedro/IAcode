"""`jarvis doctor`: what is installed, configured and downloaded — and what is missing.

Read-only: it never calls an AI model, never downloads anything and never prints a
secret (only whether each key is set).
"""

from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

from jarvis.config import PROJECT_ROOT, Settings


class Level(StrEnum):
    OK = "✓"
    WARN = "⚠"
    FAIL = "✗"


@dataclass(frozen=True)
class Check:
    area: str
    level: Level
    message: str
    hint: str = ""


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _opencode_version(binary: str) -> str | None:
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    return out.stdout.strip().removeprefix("opencode ").strip() or "?"


def run_checks(settings: Settings, *, port: int = 8300) -> list[Check]:
    import os

    checks: list[Check] = []
    add = checks.append
    version = ".".join(map(str, sys.version_info[:3]))
    add(
        Check(
            "python",
            Level.OK if sys.version_info >= (3, 12) else Level.FAIL,
            f"Python {version}",
            "" if sys.version_info >= (3, 12) else "use Python 3.12+",
        )
    )
    add(
        Check(
            "custo",
            Level.OK if settings.cost_mode.value == "FREE_ONLY" else Level.WARN,
            f"COST_MODE={settings.cost_mode.value}",
            "" if settings.cost_mode.value == "FREE_ONLY" else "agentes pagos podem ser usados",
        )
    )

    # Agents
    if settings.opencode.enabled:
        found = _opencode_version(settings.opencode.binary)
        add(
            Check(
                "opencode",
                Level.OK if found else Level.WARN,
                f"OpenCode {found}" if found else "OpenCode não encontrado",
                "" if found else "instale o OpenCode para usar os modelos grátis do Zen",
            )
        )
    keys = sorted({c.api_key_env for c in settings.openai_compat if c.enabled})
    configured = [k for k in keys if os.environ.get(k, "").strip()]
    add(
        Check(
            "chaves",
            Level.OK if configured else Level.WARN,
            f"{len(configured)} de {len(keys)} chaves de agentes online definidas"
            + (f" ({', '.join(configured)})" if configured else ""),
            "" if configured else "opcional: crie contas grátis e preencha o .env",
        )
    )

    # Optional extras
    for area, module, extra in (
        ("servidor", "fastapi", "server"),
        ("voz", "pywhispercpp", "voice"),
        ("voz", "sounddevice", "voice"),
        ("local", "mlx_lm", "local"),
    ):
        ok = _has(module)
        add(
            Check(
                area,
                Level.OK if ok else Level.WARN,
                f"{module} {'instalado' if ok else 'ausente'}",
                "" if ok else f"pip install -e '.[{extra}]'",
            )
        )

    # Downloaded models
    models = settings.data_dir / "models"
    whisper = models / "whisper" / f"ggml-{settings.voice.stt_model}.bin"
    add(
        Check(
            "voz",
            Level.OK if whisper.is_file() else Level.WARN,
            f"modelo de fala {settings.voice.stt_model}: "
            + ("baixado" if whisper.is_file() else "não baixado"),
            "" if whisper.is_file() else "baixa sozinho no 1º uso da voz (~550 MB)",
        )
    )
    if settings.voice.tts_engine == "piper" or (models / "piper").exists():
        piper = models / "piper" / f"{settings.voice.piper_voice}.onnx"
        add(
            Check(
                "voz",
                Level.OK if piper.is_file() else Level.WARN,
                f"voz Piper {settings.voice.piper_voice}: "
                + ("baixada" if piper.is_file() else "não baixada"),
                "" if piper.is_file() else "baixa sozinha no 1º uso (~60 MB)",
            )
        )
    from jarvis.voice.wakeword import is_downloaded as wake_downloaded

    wake_ok = wake_downloaded(models / "wakeword", settings.voice.wake_word_model)
    add(
        Check(
            "voz",
            Level.OK if wake_ok else Level.WARN,
            f"wake word {settings.voice.wake_word_model}: "
            + ("baixado" if wake_ok else "não baixado"),
            "" if wake_ok else "baixa sozinho no 1º uso de --wake (~9 MB)",
        )
    )
    if settings.local.enabled:
        from jarvis.agents.local.mlx import LocalModelStore

        store = LocalModelStore(models / "mlx")
        ok = store.is_downloaded(settings.local.model)
        add(
            Check(
                "local",
                Level.OK if ok else Level.WARN,
                f"modelo local {settings.local.model.split('/')[-1]}: "
                + ("baixado" if ok else "não baixado"),
                "" if ok else "jarvis local download (~2,3 GB)",
            )
        )

    # Interface
    built = (PROJECT_ROOT / "web" / "dist" / "index.html").is_file()
    add(
        Check(
            "interface",
            Level.OK if built else Level.WARN,
            "interface web compilada" if built else "interface web não compilada",
            "" if built else "cd web && npm install && npm run build",
        )
    )
    free = _port_free(port)
    add(
        Check(
            "interface",
            Level.OK if free else Level.WARN,
            f"porta {port} {'livre' if free else 'em uso'}",
            "" if free else "o JARVIS já está rodando? ou use: jarvis serve --port 8301",
        )
    )

    # Tools
    tools = settings.tools
    add(Check("ferramentas", Level.OK, "ligadas" if tools.enabled else "desligadas"))
    if tools.enabled:
        dirs = ", ".join(tools.allowed_dirs) or "nenhuma"
        add(Check("ferramentas", Level.OK, f"pastas liberadas para leitura: {dirs}"))
        sandbox = shutil.which("sandbox-exec") is not None
        add(
            Check(
                "ferramentas",
                Level.OK if sandbox else Level.WARN,
                "sandbox do macOS disponível" if sandbox else "sem sandbox: run_python desligado",
            )
        )
    return checks


def render(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        line = f"{check.level.value} {check.area:<12} {check.message}"
        if check.hint:
            line += f"\n  {'':<12} → {check.hint}"
        lines.append(line)
    problems = sum(c.level is Level.FAIL for c in checks)
    warnings = sum(c.level is Level.WARN for c in checks)
    lines.append("")
    lines.append(
        "Tudo pronto."
        if not problems and not warnings
        else f"{problems} problema(s) e {warnings} aviso(s). Os avisos são opcionais."
    )
    return "\n".join(lines)
