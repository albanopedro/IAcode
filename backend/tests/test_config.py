import pytest

from jarvis.agents.factory import build_agents
from jarvis.agents.openai_compat.adapter import OpenAICompatAgent
from jarvis.agents.opencode.adapter import OpenCodeAgent
from jarvis.config import DEFAULT_CONFIG, load_dotenv, load_settings
from jarvis.core.types import CostClass, CostMode


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("COST_MODE", "JARVIS_CONFIG", "JARVIS_TEST_KEY"):
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
    # Every OpenCode agent shares one private server.
    servers = {id(a.server) for a in agents if isinstance(a, OpenCodeAgent)}
    assert len(servers) == 1


def test_disabled_agents_are_not_built(tmp_path):
    config = tmp_path / "agents.toml"
    config.write_text(
        "[opencode]\nenabled = false\n\n"
        "[[openai_compat]]\n"
        'id = "x"\nname = "X"\nprovider = "x"\nbase_url = "https://x"\n'
        'model = "m"\napi_key_env = "X_KEY"\nenabled = false\n'
    )
    assert build_agents(load_settings(config, tmp_path / "x.env")) == []
