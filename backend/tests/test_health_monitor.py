import asyncio

from jarvis.core.agent_manager import AgentManager
from jarvis.core.cost_guard import CostGuard
from jarvis.core.health_monitor import HealthMonitor
from jarvis.core.types import Health
from tests.fakes import FakeAgent


async def test_monitor_rechecks_and_recovers_offline_agents():
    agent = FakeAgent("a", health=Health.OFFLINE)
    manager = AgentManager([agent], CostGuard())
    monitor = HealthMonitor(manager, interval=0.01)
    monitor.start()
    try:
        await asyncio.sleep(0.05)
        assert manager.status("a").health is Health.OFFLINE
        agent.health = Health.AVAILABLE  # the provider came back
        await asyncio.sleep(0.05)
        assert manager.status("a").health is Health.AVAILABLE
        assert monitor.runs >= 2
    finally:
        await monitor.stop()
    assert not monitor.running


async def test_monitor_survives_errors():
    class Exploding(AgentManager):
        async def refresh_health(self, *, force=False):
            raise RuntimeError("boom")

    monitor = HealthMonitor(Exploding([], CostGuard()), interval=0.01)
    monitor.start()
    await asyncio.sleep(0.05)
    assert monitor.running and monitor.runs >= 2
    assert monitor.last_error == "boom"
    await monitor.stop()
