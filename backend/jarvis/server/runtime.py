"""Everything the server shares between connections: agents, assistant, memory, voice."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from jarvis.agents.factory import build_agents
from jarvis.config import Settings, load_settings
from jarvis.core.agent_manager import AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.health_monitor import HealthMonitor
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.usage_store import UsageStore
from jarvis.memory.assistant import Assistant
from jarvis.memory.factory import build_assistant


@dataclass
class VoiceEngines:
    stt: object
    tts: dict[str, object] = field(default_factory=dict)


class JarvisRuntime:
    def __init__(
        self,
        settings: Settings,
        manager: AgentManager,
        assistant: Assistant | None = None,
        *,
        voice: VoiceEngines | None = None,
    ) -> None:
        self.settings = settings
        self.manager = manager
        self.assistant = assistant or Assistant(Orchestrator(manager))
        self.monitor = HealthMonitor(manager, settings.health_interval)
        self._voice = voice
        self._voice_lock = asyncio.Lock()
        self.voice_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> JarvisRuntime:
        settings = settings or load_settings()
        store = UsageStore(settings.data_dir / "jarvis.db")
        manager = AgentManager(build_agents(settings), CostGuard(settings.cost_mode), store=store)
        return cls(settings, manager, build_assistant(settings, manager))

    @property
    def memory(self):
        return self.assistant.memory

    async def start(self) -> None:
        self.monitor.start()

    async def stop(self) -> None:
        await self.monitor.stop()
        await self.assistant.wait_background()
        await self.manager.close()
        await self.assistant.close()

    # -- voice (loaded on first use: the models take a few seconds) -----------------

    @property
    def voice_ready(self) -> bool:
        return self._voice is not None

    async def wake_word(self):
        """A fresh wake-word detector (its state belongs to one audio stream)."""
        from jarvis.voice.factory import build_wake_word

        detector = build_wake_word(self.settings)
        await asyncio.to_thread(detector.warm_up)
        return detector

    async def voice(self, tts_engine: str | None = None):
        """Return (stt, tts), loading the models the first time."""
        engine = tts_engine or self.settings.voice.tts_engine
        async with self._voice_lock:
            if self._voice is None:
                from jarvis.voice.factory import build_stt

                stt = build_stt(self.settings)
                await asyncio.to_thread(stt.warm_up)
                self._voice = VoiceEngines(stt)
            if engine not in self._voice.tts:
                from jarvis.voice.factory import build_tts

                tts = build_tts(self.settings, engine)
                await asyncio.to_thread(tts.warm_up)
                self._voice.tts[engine] = tts
        return self._voice.stt, self._voice.tts[engine]
