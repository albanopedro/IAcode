"""Conversation history, owned by JARVIS (not by any agent).

Because the history lives here, the orchestrator can switch agents mid-conversation
without losing context. Phase 6 will persist it in SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.core.types import Message

DEFAULT_WINDOW = 20  # messages sent to the agent (user + assistant turns)


@dataclass
class Turn:
    message: Message
    agent_id: str | None = None


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(Message(role="user", content=text)))

    def add_assistant(self, text: str, agent_id: str) -> None:
        self.turns.append(Turn(Message(role="assistant", content=text), agent_id))

    def drop_last_user(self) -> None:
        if self.turns and self.turns[-1].message.role == "user":
            self.turns.pop()

    def window(self, size: int = DEFAULT_WINDOW) -> list[Message]:
        return [turn.message for turn in self.turns[-size:]]

    def clear(self) -> None:
        self.turns.clear()
