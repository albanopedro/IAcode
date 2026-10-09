"""A local model through MLX-LM (Apple's MLX, MIT), the offline fallback of JARVIS.

- The model is downloaded ONCE, explicitly (``jarvis local download``), into
  ``data/models/mlx``. After that, everything runs offline.
- JARVIS starts a private ``mlx_lm server`` only when the model is first needed:
  127.0.0.1, random port, no browser origin accepted (the server's default would
  accept any website), Hugging Face forced offline. It stops after some idle
  minutes to give the RAM back (~3 GB while loaded).
- The agent is ``local``: free, private, never billed. Long-term facts may be
  shared with it (it is a "private" agent for the memory policy).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

import httpx

from jarvis.core.errors import InvalidResponseError, NotConfiguredError, ProviderUnavailableError
from jarvis.core.provider import AIProvider, HealthReport
from jarvis.core.types import (
    AgentInfo,
    AIRequest,
    AIResponse,
    Capability,
    CostClass,
    Health,
    Privacy,
    TaskType,
)

NO_ORIGIN = "http://127.0.0.1:1"  # matches no real page: browsers are refused
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR")


def mlx_available() -> str | None:
    """None when MLX-LM can run here; otherwise why not."""
    if sys.platform != "darwin":
        return "MLX só roda em Macs com Apple Silicon"
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return "MLX-LM não instalado (pip install -e '.[local]')"
    return None


class LocalModelStore:
    """Where local models live, and the one-time download."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)

    def path(self, repo: str) -> Path:
        return self.models_dir / repo.replace("/", "--")

    def is_downloaded(self, repo: str) -> bool:
        path = self.path(repo)
        return (path / "config.json").is_file() and any(path.glob("*.safetensors"))

    def size_bytes(self, repo: str) -> int:
        path = self.path(repo)
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0

    def download(self, repo: str) -> Path:
        from huggingface_hub import snapshot_download

        target = self.path(repo)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"],
        )
        return target

    def remove(self, repo: str) -> bool:
        path = self.path(repo)
        if path.exists():
            shutil.rmtree(path)
            return True
        return False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MLXServer:
    def __init__(
        self,
        model_path: Path,
        *,
        max_tokens: int = 1024,
        startup_timeout: float = 180.0,
        idle_seconds: float = 600.0,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.startup_timeout = startup_timeout
        self.idle_seconds = idle_seconds
        self.url: str | None = None
        self.last_used = 0.0
        self._proc: asyncio.subprocess.Process | None = None
        self._drain: asyncio.Task | None = None
        self._idle: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def command(self, port: int) -> list[str]:
        return [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--allowed-origins",
            NO_ORIGIN,
            "--max-tokens",
            str(self.max_tokens),
            "--log-level",
            "WARNING",
        ]

    @staticmethod
    def environment() -> dict[str, str]:
        env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
        # Never touch the network at run time: the model is already on disk.
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        return env

    async def ensure_started(self) -> str:
        async with self._lock:
            self.last_used = time.monotonic()
            if self.running and self.url:
                return self.url
            await self._stop()
            port = _free_port()
            self._proc = await asyncio.create_subprocess_exec(
                *self.command(port),
                env=self.environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._drain = asyncio.create_task(self._drain_output())
            self.url = f"http://127.0.0.1:{port}"
            try:
                await self._wait_ready()
            except Exception:
                await self._stop()
                raise
            if self._idle is None or self._idle.done():
                self._idle = asyncio.create_task(self._idle_watch())
            return self.url

    async def _drain_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        with contextlib.suppress(Exception):
            while await self._proc.stdout.readline():
                pass

    async def _wait_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        async with httpx.AsyncClient(timeout=2) as client:
            while loop.time() < deadline:
                if not self.running:
                    raise ProviderUnavailableError("o servidor local do MLX terminou ao iniciar")
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(f"{self.url}/v1/models")).status_code == 200:
                        return
                await asyncio.sleep(0.5)
        raise ProviderUnavailableError("o modelo local demorou demais para carregar")

    async def _idle_watch(self) -> None:
        while self.running:
            await asyncio.sleep(min(30.0, self.idle_seconds))
            if time.monotonic() - self.last_used > self.idle_seconds:
                async with self._lock:
                    if time.monotonic() - self.last_used > self.idle_seconds:
                        await self._stop()  # give the RAM back
                return

    async def close(self) -> None:
        async with self._lock:
            await self._stop()
        if self._idle is not None:
            self._idle.cancel()

    async def _stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 10)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        if self._drain is not None:
            self._drain.cancel()
            self._drain = None
        self.url = None


class LocalMLXAgent(AIProvider):
    def __init__(
        self,
        repo: str,
        store: LocalModelStore,
        *,
        name: str | None = None,
        priority: int = 30,
        context_window: int | None = 32768,
        capabilities: frozenset[Capability] = frozenset(
            {Capability.CHAT, Capability.CODE, Capability.MATH, Capability.REASONING}
        ),
        quality: dict[TaskType, float] | None = None,
        max_tokens: int = 1024,
        idle_minutes: float = 10,
        timeout: float = 180.0,
        server: MLXServer | None = None,
    ) -> None:
        self.repo = repo
        self.store = store
        self.timeout = timeout
        self.server = server or MLXServer(
            store.path(repo), max_tokens=max_tokens, idle_seconds=idle_minutes * 60
        )
        short = repo.split("/")[-1]
        self.info = AgentInfo(
            id=f"local:{short}",
            name=name or f"Local · {short}",
            provider="mlx",
            model=repo,
            cost_class=CostClass.LOCAL,
            is_local=True,
            privacy=Privacy.LOCAL,
            capabilities=capabilities,
            priority=priority,
            context_window=context_window,
            quality=quality or {},
        )

    async def check_health(self) -> HealthReport:
        if (reason := mlx_available()) is not None:
            return HealthReport(Health.UNCONFIGURED, reason)
        if not self.store.is_downloaded(self.repo):
            return HealthReport(
                Health.UNCONFIGURED, "modelo local não baixado (rode: jarvis local download)"
            )
        # Healthy without loading it: the model only takes RAM when it is actually used.
        return HealthReport(Health.AVAILABLE, "pronto (carrega no primeiro uso)")

    async def generate(self, request: AIRequest) -> AIResponse:
        if (reason := mlx_available()) is not None:
            raise NotConfiguredError(reason)
        if not self.store.is_downloaded(self.repo):
            raise NotConfiguredError("modelo local não baixado (rode: jarvis local download)")
        started = time.perf_counter()
        url = await self.server.ensure_started()
        payload = {
            "model": str(self.store.path(self.repo)),
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_output_tokens or self.server.max_tokens,
            "temperature": 0.6,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{url}/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError("o modelo local demorou demais para responder") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"modelo local indisponível: {exc}") from exc
        finally:
            self.server.last_used = time.monotonic()
        if response.status_code >= 400:
            raise ProviderUnavailableError(f"modelo local: HTTP {response.status_code}")
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError("resposta inesperada do modelo local") from exc
        text = _THINK.sub("", text).strip()
        if not text:
            raise InvalidResponseError("o modelo local devolveu uma resposta vazia")
        usage = body.get("usage") or {}
        return AIResponse(
            text=text,
            agent_id=self.id,
            model=self.repo,
            cost=0.0,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def close(self) -> None:
        await self.server.close()
