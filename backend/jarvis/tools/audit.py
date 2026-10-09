"""Audit log of every tool call (SQLite, local). Arguments and results are truncated."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    at              REAL NOT NULL,
    conversation_id TEXT,
    agent_id        TEXT,
    tool            TEXT NOT NULL,
    args            TEXT NOT NULL,
    decision        TEXT NOT NULL,
    ok              INTEGER,
    result          TEXT,
    duration_ms     REAL
);
"""
MAX_LOGGED = 500


@dataclass(frozen=True)
class AuditEntry:
    id: int
    at: float
    conversation_id: str | None
    agent_id: str | None
    tool: str
    args: str
    decision: str
    ok: bool | None
    result: str | None
    duration_ms: float | None


class AuditLog:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def record(
        self,
        *,
        tool: str,
        args: dict,
        decision: str,
        ok: bool | None = None,
        result: str | None = None,
        duration_ms: float | None = None,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        args_text = json.dumps(args, ensure_ascii=False, default=str)[:MAX_LOGGED]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO tool_calls (at, conversation_id, agent_id, tool, args, decision, ok, "
                "result, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    conversation_id,
                    agent_id,
                    tool,
                    args_text,
                    decision,
                    None if ok is None else int(ok),
                    (result or "")[:MAX_LOGGED] if result is not None else None,
                    duration_ms,
                ),
            )

    def recent(self, limit: int = 30) -> list[AuditEntry]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, at, conversation_id, agent_id, tool, args, decision, ok, result, "
                "duration_ms FROM tool_calls ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            AuditEntry(
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                None if r[7] is None else bool(r[7]),
                r[8],
                r[9],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._db.close()
