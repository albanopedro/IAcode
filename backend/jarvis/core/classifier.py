"""Task classifier: decides what kind of request the user made.

Phase 2 uses keyword rules (PT-BR and English). It is fast, free and offline;
a model-based classifier can replace it later behind the same function.
"""

from __future__ import annotations

import re
import unicodedata

from jarvis.core.types import TaskType

_RULES: list[tuple[TaskType, list[str]]] = [
    (
        TaskType.CODE,
        [
            r"\bcodigo\b",
            r"\bcode\b",
            r"\bprogram(a|ar|acao|ming)\b",
            r"\bscript\b",
            r"\bfuncao\b",
            r"\bfunction\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\btypescript\b",
            r"\bjava\b",
            r"\brust\b",
            r"\bgolang\b",
            r"\bsql\b",
            r"\bbug\b",
            r"\bdebug",
            r"\bcompil",
            r"\bapi\b",
            r"\bregex\b",
            r"\bclasse\b",
            r"\brefator",
            r"\bstack ?trace\b",
            r"\bexception\b",
            r"\bhtml\b",
            r"\bcss\b",
        ],
    ),
    (
        TaskType.MATH,
        [
            r"\bcalcul",
            r"\bequac",
            r"\bequation\b",
            r"\bderivad",
            r"\bintegral",
            r"\bmatriz",
            r"\bmatrix\b",
            r"\bprobabilidade\b",
            r"\bestatistic",
            r"\bquanto (e|da|vale)\b",
            r"\braiz quadrada\b",
            r"\bporcentagem\b",
            r"\d+\s*[-+*/^]\s*\d+",
        ],
    ),
    (
        TaskType.RESEARCH,
        [
            r"\bpesquis",
            r"\bresearch\b",
            r"\bcompar[ae]",
            r"\bfontes?\b",
            r"\bnoticias?\b",
            r"\bnews\b",
            r"\bhistoria d[aeo]\b",
            r"\bresum[ao] sobre\b",
        ],
    ),
]

_COMPILED = [(task, [re.compile(p) for p in patterns]) for task, patterns in _RULES]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def classify(text: str) -> TaskType:
    normalized = _normalize(text)
    best, best_hits = TaskType.CHAT, 0
    for task, patterns in _COMPILED:
        hits = sum(1 for pattern in patterns if pattern.search(normalized))
        if hits > best_hits:
            best, best_hits = task, hits
    return best
