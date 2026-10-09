"""Run short Python snippets inside a macOS sandbox (sandbox-exec / Seatbelt).

Inside the sandbox the code CANNOT:
- read your files (home folder, projects, /etc…) — only Python's own library;
- write anywhere except a temporary folder that is deleted afterwards;
- open network connections;
- start other programs or fork processes.
On top of that: 10 s wall-clock limit, 5 s of CPU, 1 MB per written file, an empty
environment (no secrets), and the output is capped. Every run needs your OK.

Verified on macOS 27 (see tests). If ``sandbox-exec`` is missing, the tool is off.
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

from jarvis.tools.base import Param, Policy, Risk, Tool, ToolContext, ToolError, ToolResult

MAX_CODE_CHARS = 10_000
MAX_OUTPUT_BYTES = 8_000
WALL_SECONDS = 10.0
CPU_SECONDS = 5
FILE_BYTES = 1_000_000


def find_python() -> Path | None:
    """The real interpreter binary. Framework builds re-exec a launcher, which the sandbox
    forbids, so we point at the binary inside Python.app when it exists."""
    base = Path(sys.base_prefix)
    framework = base / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    if framework.is_file():
        return framework
    candidate = base / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    return candidate if candidate.is_file() else None


def sandbox_profile(python: Path) -> str:
    # Python may read only its own installation (standard library, the binary itself).
    roots = {str(Path(sys.base_prefix).resolve()), str(python.resolve().parent)}
    reads = " ".join(f'(subpath "{r}")' for r in sorted(roots))
    return f"""(version 1)
(deny default)
(allow process-exec (literal "{python.resolve()}"))
(allow file-read* {reads} (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share")
  (subpath "/private/var/db/timezone") (subpath "/dev") (literal "/") (literal "/private")
  (literal "/private/var"))
(allow file-read-metadata)
(allow file-read* file-write* (subpath (param "WORK")))
(allow sysctl-read)
(deny network*)
(deny process-fork)
"""


def _limits() -> None:  # runs in the child, before the interpreter starts
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()


class PythonSandboxTool(Tool):
    name = "run_python"
    title = "Executar Python (sandbox)"
    description = (
        "Executa um código Python curto numa sandbox sem internet e sem acesso aos arquivos "
        "do usuário, e devolve o que ele imprimir com print(). Use para cálculos longos, "
        "simulações ou processamento de texto. Só a biblioteca padrão está disponível."
    )
    params = (Param("code", "o código Python; use print() para mostrar o resultado"),)
    risk = Risk.DANGEROUS
    default_policy = Policy.CONFIRM
    timeout = WALL_SECONDS + 5

    def __init__(self) -> None:
        self.sandbox_exec = shutil.which("sandbox-exec")
        self.python = find_python()

    def available(self) -> str | None:
        if sys.platform != "darwin" or self.sandbox_exec is None:
            return "sandbox do macOS (sandbox-exec) indisponível"
        if self.python is None:
            return "interpretador Python não encontrado"
        return None

    def _prepare(self, tmp: str, code: str) -> tuple[Path, Path]:
        work = Path(tmp).resolve()
        (work / "snippet.py").write_text(code, encoding="utf-8")
        profile = work / "profile.sb"
        profile.write_text(sandbox_profile(self.python), encoding="utf-8")
        return work, profile

    def describe_call(self, args: dict[str, Any]) -> str:
        code = str(args.get("code", ""))
        preview = code if len(code) <= 400 else code[:400] + "…"
        return f"Executar na sandbox:\n{preview}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = self.arg(args, "code", max_len=MAX_CODE_CHARS)
        if self.available() is not None:
            raise ToolError(self.available())
        with tempfile.TemporaryDirectory(prefix="jarvis-sandbox-") as tmp:
            work, profile = await asyncio.to_thread(self._prepare, tmp, code)
            proc = await asyncio.create_subprocess_exec(
                self.sandbox_exec,
                "-f",
                str(profile),
                "-D",
                f"WORK={work}",
                str(self.python),
                "-I",
                "-S",
                "-B",
                "snippet.py",
                cwd=work,
                env={
                    "PATH": "/usr/bin",
                    "HOME": str(work),
                    "LANG": "C.UTF-8",
                    "PYTHONIOENCODING": "utf-8",
                },
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                preexec_fn=_limits,
            )
            try:
                output, _ = await asyncio.wait_for(proc.communicate(), WALL_SECONDS)
            except TimeoutError:
                proc.kill()
                await proc.communicate()  # drain and close the pipes
                return ToolResult(
                    f"tempo esgotado ({WALL_SECONDS:.0f}s): o código foi interrompido", ok=False
                )
        text = output[:MAX_OUTPUT_BYTES].decode("utf-8", "replace").strip()
        if len(output) > MAX_OUTPUT_BYTES:
            text += "\n[…saída cortada]"
        if proc.returncode == -signal.SIGXCPU:
            return ToolResult(
                f"limite de CPU ({CPU_SECONDS}s) atingido: o código foi interrompido", ok=False
            )
        if proc.returncode != 0:
            return ToolResult(
                f"o código terminou com erro (código {proc.returncode}):\n{text}", ok=False
            )
        return ToolResult(text or "(o código não imprimiu nada; use print())")
