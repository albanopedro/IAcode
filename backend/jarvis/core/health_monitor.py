"""Background health monitor for long-running sessions (chat, and later the server).

Every ``interval`` seconds it re-checks every agent. Health checks never spend
generation quota (they list models or inspect the sandbox), so this is free.
Offline agents come back by themselves when their check passes again.
"""

from __future__ import annotations

import asyncio
import contextlib

from jarvis.core.agent_manager import AgentManager

DEFAULT_INTERVAL = 120.0


class HealthMonitor:
    def __init__(self, manager: AgentManager, interval: float = DEFAULT_INTERVAL) -> None:
        self.manager = manager
        self.interval = interval
        self.runs = 0
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._loop(), name="jarvis-health-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.manager.refresh_health(force=True)
                self.last_error = None
            except Exception as exc:  # the monitor must never die
                self.last_error = str(exc)
            self.runs += 1
            await asyncio.sleep(self.interval)
