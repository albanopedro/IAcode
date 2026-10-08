"""Build the assistant (orchestrator + memory) from settings."""

from __future__ import annotations

from jarvis.config import Settings
from jarvis.core.agent_manager import AgentManager
from jarvis.core.orchestrator import Orchestrator, ShareWith
from jarvis.memory.assistant import Assistant
from jarvis.memory.store import MemoryStore


def build_assistant(settings: Settings, manager: AgentManager) -> Assistant:
    memory_cfg = settings.memory
    orchestrator = Orchestrator(manager, share_private_with=ShareWith(memory_cfg.share_facts_with))
    store = MemoryStore(settings.data_dir / "memory.db") if memory_cfg.enabled else None
    return Assistant(orchestrator, store, summarize=memory_cfg.summarize, window=memory_cfg.window)
