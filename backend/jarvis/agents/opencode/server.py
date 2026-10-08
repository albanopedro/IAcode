"""A private ``opencode serve`` process owned by JARVIS.

Why a private server instead of the user's background service:
- it listens on 127.0.0.1 only, on a random port, with a one-time password;
- it runs in an empty temporary directory, so there is nothing to read or edit;
- JARVIS controls its lifecycle and shuts it down on exit.

The OpenCode v2 HTTP API is verified at start-up against its OpenAPI document,
so a future OpenCode upgrade fails loudly instead of misbehaving.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import socket
import tempfile
from typing import Any

import httpx

from jarvis.agents.errors import error_from_message, error_from_status, parse_retry_after
from jarvis.core.errors import InvalidResponseError, NotConfiguredError, ProviderUnavailableError

REQUIRED_OPERATIONS = frozenset(
    {
        "server.info",
        "model.list",
        "provider.list",
        "provider.get",
        "session.create",
        "session.get",
        "session.prompt",
        "session.message.list",
        "session.interrupt",
        "session.remove",
    }
)
_PASSWORD = re.compile(r"server password (\S+)")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OpenCodeServer:
    def __init__(
        self,
        binary: str = "opencode",
        *,
        startup_timeout: float = 30.0,
        request_timeout: float = 60.0,
    ) -> None:
        self.binary = binary
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._drain: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.workdir: str | None = None
        self.version: str | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def ensure_started(self) -> None:
        async with self._lock:
            if self.running and self._client is not None:
                return
            await self._shutdown()
            await self._start()

    async def _start(self) -> None:
        executable = shutil.which(self.binary)
        if executable is None:
            raise NotConfiguredError(f"OpenCode binary not found: {self.binary!r}")

        self.workdir = tempfile.mkdtemp(prefix="jarvis-opencode-")
        port = _free_port()
        self._proc = await asyncio.create_subprocess_exec(
            executable,
            "serve",
            "--hostname",
            "127.0.0.1",
            "--port",
            str(port),
            cwd=self.workdir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            password = await asyncio.wait_for(self._read_password(), self.startup_timeout)
        except (TimeoutError, NotConfiguredError) as exc:
            await self._shutdown()
            raise NotConfiguredError(f"OpenCode server did not start: {exc}") from exc

        # Keep reading stdout so the pipe never fills up and blocks the server.
        self._drain = asyncio.create_task(self._drain_stdout())
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            auth=("opencode", password),
            timeout=self.request_timeout,
        )
        try:
            await self._wait_until_ready()
            await self._check_contract()
            await self._wait_for_catalog()
        except Exception:
            await self._shutdown()
            raise

    async def _read_password(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise NotConfiguredError("process exited before printing its password")
            match = _PASSWORD.search(line.decode(errors="replace"))
            if match:
                return match.group(1)

    async def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        with contextlib.suppress(Exception):
            while await self._proc.stdout.readline():
                pass

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        while True:
            try:
                info = await self.request("GET", "/api/info")
                self.version = info.get("version")
                return
            except ProviderUnavailableError:
                if loop.time() > deadline:
                    raise
                await asyncio.sleep(0.2)

    async def _check_contract(self) -> None:
        spec = await self.request("GET", "/openapi.json")
        operations = {
            op.get("operationId")
            for methods in spec.get("paths", {}).values()
            for op in methods.values()
            if isinstance(op, dict)
        }
        missing = sorted(REQUIRED_OPERATIONS - operations)
        if missing:
            raise NotConfiguredError(
                f"OpenCode {self.version} has an incompatible API (missing: {', '.join(missing)})"
            )

    async def _wait_for_catalog(self) -> None:
        """Right after start-up the provider and model snapshots are still empty."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        while loop.time() < deadline:
            providers = (await self.request("GET", "/api/provider"))["data"]
            models = (await self.request("GET", "/api/model"))["data"]
            if providers and models:
                return
            await asyncio.sleep(0.25)
        raise NotConfiguredError("OpenCode started but never listed its providers and models")

    async def request(self, method: str, path: str, json: Any = None) -> Any:
        if self._client is None:
            raise ProviderUnavailableError("OpenCode server is not running")
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"OpenCode request timed out: {path}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"OpenCode request failed: {exc}") from exc

        if response.status_code == 204 or not response.content:
            return None
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"OpenCode returned non-JSON for {path}") from exc

        if response.status_code >= 400:
            message = body.get("message", response.text) if isinstance(body, dict) else body
            retry = parse_retry_after(response.headers.get("retry-after"))
            if response.status_code in (400, 503):
                # OpenCode wraps provider errors; read the message to classify them.
                raise error_from_message(str(message), retry)
            raise error_from_status(response.status_code, str(message), retry)
        return body

    async def models(self) -> list[dict]:
        return (await self.request("GET", "/api/model"))["data"]

    async def provider(self, provider_id: str) -> dict:
        return (await self.request("GET", f"/api/provider/{provider_id}"))["data"]

    async def close(self) -> None:
        async with self._lock:
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        if self._drain is not None:
            self._drain.cancel()
            self._drain = None
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None
