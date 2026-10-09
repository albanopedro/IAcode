"""Build the agent list from settings."""

from __future__ import annotations

from jarvis.agents.openai_compat.adapter import OpenAICompatAgent
from jarvis.agents.opencode.adapter import OpenCodeAgent
from jarvis.agents.opencode.runtime import OpenCodeRuntime
from jarvis.config import Settings
from jarvis.core.provider import AIProvider
from jarvis.core.types import AgentInfo


def build_agents(settings: Settings) -> list[AIProvider]:
    agents: list[AIProvider] = []

    if settings.opencode.enabled and settings.opencode.models:
        runtime = OpenCodeRuntime(settings.opencode.binary)
        for model in settings.opencode.models:
            agents.append(
                OpenCodeAgent(
                    runtime,
                    model.id,
                    name=model.name,
                    priority=model.priority,
                    capabilities=model.capabilities,
                    privacy=model.privacy,
                    quality=model.quality,
                    rpm_limit=model.rpm_limit,
                    timeout=settings.opencode.timeout,
                )
            )

    if settings.local.enabled:
        from jarvis.agents.local.mlx import LocalMLXAgent, LocalModelStore

        local = settings.local
        agents.append(
            LocalMLXAgent(
                local.model,
                LocalModelStore(settings.data_dir / "models" / "mlx"),
                priority=local.priority,
                context_window=local.context_window,
                max_tokens=local.max_tokens,
                idle_minutes=local.idle_minutes,
            )
        )

    for cfg in settings.openai_compat:
        if not cfg.enabled:
            continue
        info = AgentInfo(
            id=cfg.id,
            name=cfg.name,
            provider=cfg.provider,
            model=cfg.model,
            cost_class=cfg.cost_class,
            capabilities=cfg.capabilities,
            priority=cfg.priority,
            privacy=cfg.privacy,
            daily_limit=cfg.daily_limit,
            rpm_limit=cfg.rpm_limit,
            quota_group=cfg.quota_group,
            context_window=cfg.context_window,
            quality=cfg.quality,
            allow_paid=cfg.allow_paid,
        )
        agents.append(
            OpenAICompatAgent(
                info,
                base_url=cfg.base_url,
                api_key_env=cfg.api_key_env,
                free_model_pattern=cfg.free_model_pattern,
                health_check=cfg.health_check,
            )
        )
    return agents
