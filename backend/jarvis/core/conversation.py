"""Conversation history, owned by JARVIS (not by any agent).

Because the history lives here, the orchestrator can switch agents mid-conversation
without losing context.

Short-term memory: only the most recent ``window`` messages go to the agent. Older
messages are folded into ``summary`` (see ``jarvis.memory.summarizer``), so long
conversations keep their thread without growing the prompt forever.
``jarvis.memory.store.PersistentConversation`` saves every change to SQLite.

Private turns (exchanged in "local only" mode) never reach a non-local agent: they are
left out of the context of any later request that is not local-only, and out of the
summaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from jarvis.core.types import Message

DEFAULT_WINDOW = 20  # messages sent to the agent (user + assistant turns)


@dataclass
class Turn:
    message: Message
    agent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    private: bool = False  # said in local-only mode: must never leave this Mac


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    id: str | None = None
    title: str = ""
    summary: str | None = None
    summarized: int = 0  # how many of the oldest turns the summary covers
    generation: int = 0  # bumps on clear, so late summaries of old turns are discarded

    def add_user(self, text: str, *, private: bool = False) -> None:
        self.turns.append(Turn(Message(role="user", content=text), private=private))
        if not self.title:
            self.title = "🔒 conversa privada" if private else text[:60]

    def add_assistant(self, text: str, agent_id: str, *, private: bool = False) -> None:
        self.turns.append(Turn(Message(role="assistant", content=text), agent_id, private=private))

    def drop_last_user(self) -> None:
        if self.turns and self.turns[-1].message.role == "user":
            self.turns.pop()

    def window(self, size: int = DEFAULT_WINDOW) -> list[Message]:
        return [turn.message for turn in self.turns[-size:]]

    def context(
        self, size: int = DEFAULT_WINDOW, *, include_private: bool = False
    ) -> list[Message]:
        """What the agent sees: the summary of older turns (if any) + the recent window.

        Private turns are included only for local-only requests.
        """
        visible = [t for t in self.turns if include_private or not t.private]
        messages = [turn.message for turn in visible[-size:]]
        hidden = len(visible) - len(messages)
        if self.summary and hidden > 0:
            note = (
                "Resumo da parte anterior desta conversa (mensagens mais antigas que as "
                f"abaixo):\n{self.summary}"
            )
            return [Message(role="system", content=note), *messages]
        return messages

    def set_summary(self, summary: str, summarized: int, generation: int) -> bool:
        if generation != self.generation or summarized <= self.summarized:
            return False
        self.summary = summary
        self.summarized = summarized
        return True

    def clear(self) -> None:
        self.turns.clear()
        self.summary = None
        self.summarized = 0
        self.title = ""
        self.generation += 1
