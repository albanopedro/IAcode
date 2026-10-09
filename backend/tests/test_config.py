import pytest

from jarvis.agents.factory import build_agents
from jarvis.agents.openai_compat.adapter import OpenAICompatAgent
from jarvis.agents.opencode.adapter import OpenCodeAgent
from jarvis.config import DEFAULT_CONFIG, load_dotenv, load_settings
from jarvis.core.types import CostClass, CostMode


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("COST_MODE", "JARVIS_CONFIG", "JARVIS_TEST_KEY", "JARVIS_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)


def test_cost_mode_defaults_to_free_only(tmp_path):
    settings = load_settings(DEFAULT_CONFIG, env_path=tmp_path / "missing.env")
    assert settings.cost_mode is CostMode.FREE_ONLY


@pytest.mark.parametrize("value", ["allow-paid", "yes", "paid", "free"])
def test_invalid_cost_mode_falls_back_to_free_only(monkeypatch, tmp_path, value):
    monkeypatch.setenv("COST_MODE", value)
    assert load_settings(DEFAULT_CONFIG, tmp_path / "x.env").cost_mode is CostMode.FREE_ONLY


def test_config_file_cannot_change_cost_mode(tmp_path):
    config = tmp_path / "agents.toml"
    config.write_text('cost_mode = "ALLOW_PAID"\n')
    assert load_settings(config, tmp_path / "x.env").cost_mode is CostMode.FREE_ONLY


def test_dotenv_never_overrides_real_environment(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nJARVIS_TEST_KEY='from-file'\nCOST_MODE=FREE_ONLY\n")
    monkeypatch.setenv("JARVIS_TEST_KEY", "from-env")
    load_dotenv(env)
    import os

    assert os.environ["JARVIS_TEST_KEY"] == "from-env"


def test_project_config_builds_only_free_agents(tmp_path):
    settings = load_settings(DEFAULT_CONFIG, tmp_path / "x.env")
    agents = build_agents(settings)
    assert any(isinstance(a, OpenCodeAgent) for a in agents)
    assert any(isinstance(a, OpenAICompatAgent) for a in agents)
    assert all(a.info.cost_class is not CostClass.PAID for a in agents)
    assert all(not a.info.allow_paid for a in agents)
    # Every OpenCode agent shares one runtime (sandbox + private server).
    runtimes = {id(a.runtime) for a in agents if isinstance(a, OpenCodeAgent)}
    assert len(runtimes) == 1


def test_disabled_agents_are_not_built(tmp_path):
    config = tmp_path / "agents.toml"
    config.write_text(
        "[opencode]\nenabled = false\n\n"
        "[[openai_compat]]\n"
        'id = "x"\nname = "X"\nprovider = "x"\nbase_url = "https://x"\n'
        'model = "m"\napi_key_env = "X_KEY"\nenabled = false\n'
    )
    assert build_agents(load_settings(config, tmp_path / "x.env")) == []


def test_project_config_declares_limits_and_groups(tmp_path):
    settings = load_settings(DEFAULT_CONFIG, tmp_path / "x.env")
    agents = {a.id: a for a in build_agents(settings)}
    openrouter = [a for a in agents.values() if a.info.provider == "openrouter"]
    assert len(openrouter) >= 2
    assert {a.info.quota_group for a in openrouter} == {"openrouter-free"}
    assert all(a.info.rpm_limit for a in openrouter)
    assert all(a.free_model_pattern == ":free$" for a in openrouter)
    assert agents["cloudflare:llama-3.1-8b"].health_check == "key"
    assert settings.health_interval > 0


def test_data_dir_can_be_set_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "data"))
    assert load_settings(DEFAULT_CONFIG, tmp_path / "x.env").data_dir == tmp_path / "data"


def test_project_config_wires_memory_and_tools_into_the_assistant(tmp_path):
    from jarvis.core.agent_manager import AgentManager
    from jarvis.core.cost_guard import CostGuard
    from jarvis.memory.factory import build_assistant

    settings = load_settings(DEFAULT_CONFIG, tmp_path / "x.env")
    settings.data_dir = tmp_path
    assistant = build_assistant(settings, AgentManager([], CostGuard()))
    assert assistant.memory is not None
    assert assistant.tools is not None
    prompt = assistant.tools.prompt()
    assert "calculator" in prompt and "```tool" in prompt
    assert "read_file" not in prompt  # no allowed_dirs configured: file tools are not offered
