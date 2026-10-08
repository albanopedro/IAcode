"""Real call to a free OpenCode Zen model (cost $0).

Skipped by default. Run with:  JARVIS_LIVE_TESTS=1 pytest -m live
"""

import os
import shutil

import pytest

from jarvis.agents.opencode.adapter import OpenCodeAgent
from jarvis.agents.opencode.runtime import OpenCodeRuntime
from jarvis.core.agent_manager import AgentManager
from jarvis.core.conversation import Conversation
from jarvis.core.cost_guard import CostGuard
from jarvis.core.orchestrator import Orchestrator

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("JARVIS_LIVE_TESTS") != "1", reason="live test"),
    pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode not installed"),
]


async def test_opencode_answers_for_free_and_keeps_context():
    runtime = OpenCodeRuntime()
    manager = AgentManager([OpenCodeAgent(runtime, "longcat-2.5-preview-free")], CostGuard())
    orchestrator = Orchestrator(manager)
    conversation = Conversation()
    try:
        first = await orchestrator.ask(conversation, "Meu nome fictício é Zeca. Diga só 'ok'.")
        second = await orchestrator.ask(conversation, "Qual é o meu nome? Responda só o nome.")
        root = runtime.sandbox.root
    finally:
        await manager.close()

    assert first.response.cost in (0, None) and second.response.cost in (0, None)
    assert "zeca" in second.response.text.lower()
    # The sandbox (own OpenCode home + empty work folder) is removed on close.
    assert not os.path.exists(root)  # noqa: ASYNC240
