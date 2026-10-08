"""An isolated OpenCode home for JARVIS.

OpenCode is a coding agent with real tools (read, edit, shell…). OpenCode Zen
only serves its free tier to the real OpenCode agent with those tools declared,
so JARVIS cannot remove them. Instead, everything around the agent is locked:

- **Own OpenCode home** (XDG config/data/state in a temporary folder): your
  credentials, history and global config are never visible, so the only
  provider is OpenCode Zen with its anonymous ``public`` key — nothing to bill.
- **Minimal environment**: only an allowlist of variables is passed on, so
  provider keys (``OPENAI_API_KEY``…) and ``OPENCODE_*`` overrides from your
  shell never reach OpenCode.
- **Empty working directory**, also set as ``PWD`` (OpenCode reads ``PWD``,
  not the process cwd, to pick its project folder).
- **Every tool needs approval** (``ask`` for everything, as the last rule). A
  non-interactive ``opencode run`` cannot ask, so OpenCode rejects each tool call
  itself. ``--auto`` (which would approve them) is never passed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

AGENT_NAME = "jarvis"
ASK_ALL = {"action": "*", "resource": "*", "effect": "ask"}
AGENT_CONFIG = {
    "agents": {
        AGENT_NAME: {
            "description": "JARVIS conversation agent. Answers with text only.",
            "mode": "primary",
            "permissions": [ASK_ALL],
        }
    },
    "default_agent": AGENT_NAME,
    "autoupdate": False,
    "share": "disabled",
}

# Variables OpenCode needs to run; everything else from the parent is dropped.
ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")


class OpenCodeSandbox:
    def __init__(self, root: str | None = None) -> None:
        self._owns_root = root is None
        self.root = Path(root or tempfile.mkdtemp(prefix="jarvis-opencode-"))
        self.workdir = self.root / "work"
        for sub in ("config/opencode", "data", "state", "work"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        config = self.root / "config" / "opencode" / "opencode.json"
        config.write_text(json.dumps(AGENT_CONFIG, indent=2), encoding="utf-8")

    def env(self, **extra: str) -> dict[str, str]:
        env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
        env.update(
            XDG_CONFIG_HOME=str(self.root / "config"),
            XDG_DATA_HOME=str(self.root / "data"),
            XDG_STATE_HOME=str(self.root / "state"),
            PWD=str(self.workdir),
        )
        env.update(extra)
        return env

    def cleanup(self) -> None:
        if self._owns_root:
            shutil.rmtree(self.root, ignore_errors=True)


def agent_is_locked(permissions: list[dict]) -> bool:
    """True when the effective rules end with "ask for everything".

    OpenCode evaluates the rules in order and the last match wins, so a final
    ``* / * / ask`` overrides every default ``allow`` that comes before it.
    """
    return bool(permissions) and permissions[-1] == ASK_ALL
