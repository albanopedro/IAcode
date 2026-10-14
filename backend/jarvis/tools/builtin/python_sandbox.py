"""Run short Python snippets inside a macOS sandbox (sandbox-exec / Seatbelt).

Inside the sandbox the code CANNOT:
- read your files (home folder, projects, /etc…) — only Python's own library;
- write anywhere except a temporary folder that is deleted afterwards;
- open network connections;
- start other programs or fork processes.
On top of that: 10 s wall-clock limit, 5 s of CPU, 1 MB per written file, 512 MB of memory,
an empty environment (no secrets), and the output is capped (the code is stopped once it
prints more than the cap). Every run needs your OK, and the confirmation shows ALL of the
code that will run.

macOS does not enforce RLIMIT_AS/RLIMIT_DATA (setting them fails), so memory is watched
from outside: the parent reads the child's resident size a few times a second and kills it.

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
MAX_MEMORY_KB = 512 * 1024
MEMORY_POLL_SECONDS = 0.2
READ_CHUNK = 64 * 1024
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


async def _resident_kb(pid: int) -> int | None:
    """Resident memory of ``pid`` in KB (None when it is gone or cannot be read)."""
    ps = await asyncio.create_subprocess_exec(
        "/bin/ps",
        "-o",
        "rss=",
        "-p",
        str(pid),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await ps.communicate()
    try:
        return int(out.split()[0])
    except (ValueError, IndexError):
        return None


async def _collect(proc: asyncio.subprocess.Process) -> tuple[bytes, bool]:
    """Read the output up to the cap, then wait for the exit.

    Past the cap the code is stopped: reading everything first would let a ``print`` loop fill
    the memory of JARVIS itself. Returns (output up to the cap, whether it was cut).
    """
    assert proc.stdout is not None
    chunks: list[bytes] = []
    size = 0
    while chunk := await proc.stdout.read(READ_CHUNK):
        if size < MAX_OUTPUT_BYTES:
            chunks.append(chunk[: MAX_OUTPUT_BYTES - size])
        size += len(chunk)
        if size > MAX_OUTPUT_BYTES:
            proc.kill()
            await proc.wait()
            return b"".join(chunks), True
    await proc.wait()
    return b"".join(chunks), False


class _MemoryWatch:
    """Kills the process when its resident memory passes ``MAX_MEMORY_KB``."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.exceeded = False
        self._proc = proc
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while self._proc.returncode is None:
            await asyncio.sleep(MEMORY_POLL_SECONDS)
            used = await _resident_kb(self._proc.pid)
            if used is not None and used > MAX_MEMORY_KB and self._proc.returncode is None:
                self.exceeded = True
                self._proc.kill()
                return

    async def stop(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


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
        # The whole code (at most MAX_CODE_CHARS): what you approve is what runs.
        return f"Executar na sandbox:\n{args.get('code', '')}"

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
            memory = _MemoryWatch(proc)
            try:
                output, truncated = await asyncio.wait_for(_collect(proc), WALL_SECONDS)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    f"tempo esgotado ({WALL_SECONDS:.0f}s): o código foi interrompido", ok=False
                )
            finally:
                if proc.returncode is None:  # cancelled from outside: never leave it running
                    proc.kill()
                await memory.stop()
        if memory.exceeded:
            return ToolResult(
                f"limite de memória ({MAX_MEMORY_KB // 1024} MB) atingido: "
                "o código foi interrompido",
                ok=False,
            )
        text = output.decode("utf-8", "replace").strip()
        if truncated:
            return ToolResult(
                f"{text}\n[…saída cortada em {MAX_OUTPUT_BYTES} bytes: o código foi interrompido]"
            )
        if proc.returncode == -signal.SIGXCPU:
            return ToolResult(
                f"limite de CPU ({CPU_SECONDS}s) atingido: o código foi interrompido", ok=False
            )
        if proc.returncode != 0:
            return ToolResult(
                f"o código terminou com erro (código {proc.returncode}):\n{text}", ok=False
            )
        return ToolResult(text or "(o código não imprimiu nada; use print())")
