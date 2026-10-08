"""Persistent usage and agent state (SQLite, standard library only).

Why it exists:
- daily quotas (e.g. OpenRouter's 50 requests/day) must survive a restart,
  otherwise JARVIS would forget how much of the day's quota it already used;
- cooldowns survive a restart, so JARVIS does not hammer a rate-limited agent;
- an agent blocked for a cost or billing problem STAYS blocked until the user
  explicitly runs ``jarvis unblock <id>`` — money safety must not reset itself.

Only counters and states are stored here: no prompts, answers or secrets.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    agent_id  TEXT NOT NULL,
    day       TEXT NOT NULL,
    requests  INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    failures  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, day)
);
CREATE TABLE IF NOT EXISTS agent_state (
    agent_id       TEXT PRIMARY KEY,
    cooldown_until REAL,
    blocked_reason TEXT,
    last_error     TEXT,
    updated_at     REAL NOT NULL
);
"""


@dataclass(frozen=True)
class StoredState:
    requests_today: int = 0
    successes_today: int = 0
    failures_today: int = 0
    cooldown_until: float | None = None
    blocked_reason: str | None = None
    last_error: str | None = None


class UsageStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def load(self, agent_id: str, day: str) -> StoredState:
        with self._lock:
            usage = self._db.execute(
                "SELECT requests, successes, failures FROM usage WHERE agent_id = ? AND day = ?",
                (agent_id, day),
            ).fetchone()
            state = self._db.execute(
                "SELECT cooldown_until, blocked_reason, last_error FROM agent_state "
                "WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        requests, successes, failures = usage or (0, 0, 0)
        cooldown, blocked, last_error = state or (None, None, None)
        return StoredState(requests, successes, failures, cooldown, blocked, last_error)

    def record_call(self, agent_id: str, day: str, *, ok: bool) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO usage (agent_id, day, requests, successes, failures) "
                "VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT (agent_id, day) DO UPDATE SET "
                "requests = requests + 1, successes = successes + excluded.successes, "
                "failures = failures + excluded.failures",
                (agent_id, day, int(ok), int(not ok)),
            )

    def save_state(
        self,
        agent_id: str,
        *,
        now: float,
        cooldown_until: float | None,
        blocked_reason: str | None,
        last_error: str | None,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO agent_state "
                "(agent_id, cooldown_until, blocked_reason, last_error, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (agent_id) DO UPDATE SET cooldown_until = excluded.cooldown_until, "
                "blocked_reason = excluded.blocked_reason, last_error = excluded.last_error, "
                "updated_at = excluded.updated_at",
                (agent_id, cooldown_until, blocked_reason, last_error, now),
            )

    def blocked(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT agent_id, blocked_reason FROM agent_state "
                "WHERE blocked_reason IS NOT NULL ORDER BY agent_id"
            ).fetchall()
        return dict(rows)

    def unblock(self, agent_id: str) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE agent_state SET blocked_reason = NULL, cooldown_until = NULL "
                "WHERE agent_id = ? AND blocked_reason IS NOT NULL",
                (agent_id,),
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._db.close()
