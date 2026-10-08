"""Turn an assistant answer into something pleasant to hear.

Answers often contain Markdown (code blocks, bullets, links) that sounds awful
when read aloud. ``speakable`` cleans it; ``split_sentences`` cuts the text so
the first sentence can be spoken while the next ones are still being synthesized.
"""

from __future__ import annotations

import re

_CODE_BLOCK = re.compile(r"```.*?(```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_TABLE_RULE = re.compile(r"^\s*\|?[\s:-]+\|[\s|:-]*$", re.MULTILINE)
_SPACES = re.compile(r"[ \t]+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")

CODE_PLACEHOLDER = " (o código está na tela) "
LINK_PLACEHOLDER = "um link"


def speakable(text: str) -> str:
    text = _CODE_BLOCK.sub(CODE_PLACEHOLDER, text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(LINK_PLACEHOLDER, text)
    text = _TABLE_RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = text.replace("|", ", ")
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    # Each bullet/line becomes its own sentence so the voice pauses between them.
    joined = "\n".join(
        line if line.endswith((".", "!", "?", ":", ";", ",")) else f"{line}."
        for line in lines
        if line
    )
    return joined.strip()


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split into sentences; very long sentences are cut at commas or spaces."""
    pieces = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    result: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(",", 0, max_chars)
            if cut < max_chars // 3:
                cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            result.append(piece[: cut + 1].strip())
            piece = piece[cut + 1 :].strip()
        if piece:
            result.append(piece)
    return result
