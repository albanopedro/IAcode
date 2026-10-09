"""SQLite storage for conversations and long-term facts (standard library only)."""

from __future__ import annotations

import sqlite3
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from jarvis.core.conversation import Conversation, Turn
from jarvis.core.types import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    summary    TEXT,
    summarized INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    agent_id        TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    done_at    REAL
);
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    normalized TEXT NOT NULL UNIQUE,
    source     TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL
);
"""

MAX_FACT_CHARS = 300


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    created_at: float


@dataclass(frozen=True)
class Task:
    id: int
    text: str
    done: bool
    created_at: float
    done_at: float | None


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: int


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock, self._db:
            return self._db.execute(sql, params)

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # -- conversations -------------------------------------------------------------

    def new_conversation(self) -> PersistentConversation:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        self._write(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (conversation_id, now, now),
        )
        return PersistentConversation(self, id=conversation_id)

    def load_conversation(self, conversation_id: str) -> PersistentConversation | None:
        rows = self._read(
            "SELECT title, summary, summarized FROM conversations WHERE id = ?", (conversation_id,)
        )
        if not rows:
            return None
        title, summary, summarized = rows[0]
        messages = self._read(
            "SELECT role, content, agent_id, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        turns = [
            Turn(Message(role=role, content=content), agent_id, created_at)
            for role, content, agent_id, created_at in messages
        ]
        return PersistentConversation(
            self,
            id=conversation_id,
            turns=turns,
            title=title,
            summary=summary,
            summarized=summarized,
        )

    def latest_conversation(self) -> PersistentConversation | None:
        rows = self._read("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1")
        return self.load_conversation(rows[0][0]) if rows else None

    def list_conversations(self, limit: int = 30) -> list[ConversationInfo]:
        rows = self._read(
            "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id GROUP BY c.id "
            "HAVING COUNT(m.id) > 0 ORDER BY c.updated_at DESC LIMIT ?",
            (limit,),
        )
        return [ConversationInfo(*row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        return (
            self._write("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0
        )

    def delete_all_conversations(self) -> int:
        return self._write("DELETE FROM conversations").rowcount

    def _append(self, conversation: Conversation, turn: Turn) -> None:
        self._write(
            "INSERT INTO messages (conversation_id, role, content, agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                conversation.id,
                turn.message.role,
                turn.message.content,
                turn.agent_id,
                turn.created_at,
            ),
        )
        self._touch(conversation)

    def _drop_last(self, conversation: Conversation) -> None:
        self._write(
            "DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages "
            "WHERE conversation_id = ?)",
            (conversation.id,),
        )

    def _clear(self, conversation: Conversation) -> None:
        self._write("DELETE FROM messages WHERE conversation_id = ?", (conversation.id,))
        self._touch(conversation)

    def _touch(self, conversation: Conversation) -> None:
        self._write(
            "UPDATE conversations SET title = ?, summary = ?, summarized = ?, updated_at = ? "
            "WHERE id = ?",
            (
                conversation.title,
                conversation.summary,
                conversation.summarized,
                time.time(),
                conversation.id,
            ),
        )

    # -- long-term facts ----------------------------------------------------------

    def add_fact(self, text: str, source: str = "user") -> Fact | None:
        """Save a fact. Returns None when the same fact is already saved."""
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        key = normalize(text)
        if not key:
            return None
        now = time.time()
        cursor = self._write(
            "INSERT OR IGNORE INTO facts (text, normalized, source, created_at) "
            "VALUES (?, ?, ?, ?)",
            (text, key, source, now),
        )
        return Fact(cursor.lastrowid, text, now) if cursor.rowcount else None

    def facts(self) -> list[Fact]:
        rows = self._read("SELECT id, text, created_at FROM facts ORDER BY id")
        return [Fact(*row) for row in rows]

    def find_facts(self, query: str) -> list[Fact]:
        key = normalize(query)
        if not key:
            return []
        return [f for f in self.facts() if key in normalize(f.text)]

    def delete_fact(self, fact_id: int) -> bool:
        return self._write("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0

    def delete_all_facts(self) -> int:
        return self._write("DELETE FROM facts").rowcount

    # -- tasks --------------------------------------------------------------------

    def add_task(self, text: str) -> Task:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        now = time.time()
        cursor = self._write("INSERT INTO tasks (text, created_at) VALUES (?, ?)", (text, now))
        return Task(cursor.lastrowid, text, False, now, None)

    def tasks(self, *, include_done: bool = False) -> list[Task]:
        where = "" if include_done else "WHERE done = 0"
        rows = self._read(
            f"SELECT id, text, done, created_at, done_at FROM tasks {where} ORDER BY id"
        )
        return [Task(r[0], r[1], bool(r[2]), r[3], r[4]) for r in rows]

    def complete_task(self, task_id: int) -> bool:
        cursor = self._write(
            "UPDATE tasks SET done = 1, done_at = ? WHERE id = ? AND done = 0",
            (time.time(), task_id),
        )
        return cursor.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        return self._write("DELETE FROM tasks WHERE id = ?", (task_id,)).rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._db.close()


class PersistentConversation(Conversation):
    """A conversation that writes every change to the memory store."""

    def __init__(self, store: MemoryStore, **fields) -> None:
        super().__init__(**fields)
        self._store = store

    def add_user(self, text: str) -> None:
        super().add_user(text)
        self._store._append(self, self.turns[-1])

    def add_assistant(self, text: str, agent_id: str) -> None:
        super().add_assistant(text, agent_id)
        self._store._append(self, self.turns[-1])

    def drop_last_user(self) -> None:
        if self.turns and self.turns[-1].message.role == "user":
            super().drop_last_user()
            self._store._drop_last(self)

    def set_summary(self, summary: str, summarized: int, generation: int) -> bool:
        changed = super().set_summary(summary, summarized, generation)
        if changed:
            self._store._touch(self)
        return changed

    def clear(self) -> None:
        super().clear()
        self._store._clear(self)
