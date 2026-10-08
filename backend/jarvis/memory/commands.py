"""Memory commands understood locally, without asking any agent (no quota, no privacy cost).

    "JARVIS, lembre que eu prefiro respostas curtas."   → saves a fact
    "Esqueça que eu prefiro respostas curtas."          → deletes matching facts
    "O que você sabe sobre mim?"                        → lists the facts

Only explicit requests are understood; JARVIS never decides on its own to
remember something about you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class CommandKind(StrEnum):
    REMEMBER = "remember"
    FORGET = "forget"
    FORGET_ALL = "forget_all"
    LIST = "list"


@dataclass(frozen=True)
class MemoryCommand:
    kind: CommandKind
    argument: str = ""


_PREFIX = r"^\s*(?:(?:ei|ok|olá|ola|oi)[\s,!]+)?(?:jarvis[\s,!.:;-]*)?(?:por\s+favor[\s,]*)?"
_REMEMBER = re.compile(
    _PREFIX
    + r"(?:lembr[ae](?:-se)?|memoriz[ae]|guard[ae]|anot[ae])[\s:,]+"
    + r"(?:(?:de\s+que|disso|isso|de|que)\b[\s:,]*)?(?P<fact>.+?)[\s.!]*$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET = re.compile(
    _PREFIX
    + r"(?:esque[cç]a|apague|remova|delete)[\s:,]+"
    + r"(?:(?:de\s+que|da\s+mem[oó]ria|o\s+fato|que|de)\b[\s:,]*)?(?P<what>.+?)[\s.!]*$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET_ALL = re.compile(r"^(?:tudo|todas?\s+as\s+mem[oó]rias|tudo\s+sobre\s+mim)$", re.IGNORECASE)
_LIST = re.compile(
    _PREFIX
    + r"(?:o\s+)?(?:que|quais)\s+(?:(?:voc[eê]|vc)\s+)?(?:sabe|lembra|memorizou|guardou|"
    + r"s[aã]o\s+(?:as\s+)?suas\s+mem[oó]rias|mem[oó]rias\s+voc[eê]\s+tem)"
    + r"(?:\s+(?:sobre|de)\s+mim)?\s*\??\s*$",
    re.IGNORECASE,
)


def parse(text: str) -> MemoryCommand | None:
    text = text.strip()
    if _LIST.match(text):
        return MemoryCommand(CommandKind.LIST)
    if match := _FORGET.match(text):
        what = match.group("what").strip()
        if _FORGET_ALL.match(what):
            return MemoryCommand(CommandKind.FORGET_ALL)
        return MemoryCommand(CommandKind.FORGET, what)
    # "Você lembra de mim?" is a question, not a request to remember something.
    if not text.endswith("?") and (match := _REMEMBER.match(text)):
        fact = match.group("fact").strip()
        return MemoryCommand(CommandKind.REMEMBER, fact) if len(fact) >= 3 else None
    return None
