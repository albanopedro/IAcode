"""Memory commands understood locally, without asking any agent (no quota, no privacy cost).

    "JARVIS, lembre que eu prefiro respostas curtas."   → saves a fact
    "Esqueça que eu prefiro respostas curtas."          → deletes matching facts
    "O que você sabe sobre mim?"                        → lists the facts

Only explicit requests are understood; JARVIS never decides on its own to
remember something about you. Verbs that also mean other things ("guarde", "anote",
"apague", "remova", "delete") count only when the sentence says it is about the memory
("guarde que…", "apague da memória…"); otherwise it is an ordinary request and goes to
the agent ("remova as duplicatas desta lista", "guarde esse código em um arquivo").
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
    # False when the sentence never said it was about the memory ("esqueça as duplicatas"):
    # if nothing matches, it is probably an ordinary request and should reach the agent.
    explicit: bool = True


_PREFIX = r"^\s*(?:(?:ei|ok|olá|ola|oi)[\s,!]+)?(?:jarvis[\s,!.:;-]*)?(?:por\s+favor[\s,]*)?"
_REMEMBER = re.compile(
    _PREFIX
    + r"(?:"
    # Verbs that are about memory by nature.
    + r"(?:lembr[ae](?:-se)?|memoriz[ae])[\s:,]+(?:(?:de\s+que|disso|isso|de|que)\b[\s:,]*)?"
    # "guarde" / "anote" also mean "save this file" / "write this down": they need
    # "que", "de que" or a colon ("guarde isso: …", "anota aí: …", "guarde na memória: …").
    + r"|(?:guard[ae]|anot[ae])(?:\s+na\s+mem[oó]ria)?"
    + r"(?:(?:\s+(?:isso|disso|a[ií]))?\s*:|[\s,]+(?:de\s+)?que\b)[\s:,]*"
    + r")(?P<fact>.+?)[\s.!]*$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET = re.compile(
    _PREFIX
    + r"(?P<verb>esque[cç]a|apague|remova|delete)[\s:,]+"
    + r"(?:(?P<marker>de\s+que|da\s+mem[oó]ria|das\s+lembran[cç]as|o\s+fato|a\s+lembran[cç]a|que)"
    + r"\b[\s:,]*)?(?P<what>.+?)[\s.!]*$",
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
        marker = match.group("marker")
        # "esqueça" is about memory by nature; "apague/remova/delete" need the marker.
        if marker is None and match.group("verb").lower() not in ("esqueça", "esqueca"):
            return None
        return MemoryCommand(CommandKind.FORGET, what, explicit=marker is not None)
    # "Você lembra de mim?" is a question, not a request to remember something.
    if not text.endswith("?") and (match := _REMEMBER.match(text)):
        fact = match.group("fact").strip()
        return MemoryCommand(CommandKind.REMEMBER, fact) if len(fact) >= 3 else None
    return None
