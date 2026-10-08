"""Real call to a free OpenCode Zen model (cost $0).

Skipped by default. Run with:  JARVIS_LIVE_TESTS=1 pytest -m live
"""

import os
import shutil

import pytest

from jarvis.agents.opencode.adapter import OpenCodeAgent
from jarvis.agents.opencode.server import OpenCodeServer
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
    server = OpenCodeServer()
    manager = AgentManager([OpenCodeAgent(server, "space-bunny-free")], CostGuard())
    orchestrator = Orchestrator(manager)
    conversation = Conversation()
    try:
        first = await orchestrator.ask(conversation, "Meu nome fictício é Zeca. Diga só 'ok'.")
        second = await orchestrator.ask(conversation, "Qual é o meu nome? Responda só o nome.")
        workdir = server.workdir
    finally:
        await manager.close()

    assert first.response.cost == 0 and second.response.cost == 0
    assert "zeca" in second.response.text.lower()
    # The temporary workspace is removed on close.
    assert workdir is not None and not os.path.exists(workdir)  # noqa: ASYNC240
