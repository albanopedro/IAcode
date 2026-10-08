"""Read the quota a provider reports in its HTTP response headers.

Providers use different conventions; all are mapped to ``RateLimitInfo``:
- Groq / OpenAI style: ``x-ratelimit-remaining-requests``,
  ``x-ratelimit-limit-requests``, ``x-ratelimit-reset-requests`` ("2m59.56s", "120ms");
- OpenRouter style: ``x-ratelimit-remaining``, ``x-ratelimit-limit``,
  ``x-ratelimit-reset`` (a Unix timestamp in milliseconds);
- IETF draft style: ``ratelimit-remaining``, ``ratelimit-limit``, ``ratelimit-reset`` (seconds).
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping

from jarvis.core.types import RateLimitInfo

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_duration(value: str) -> float | None:
    """'2m59.56s' → 179.56, '120ms' → 0.12, '14h12m' → 51120, '30' → 30."""
    value = value.strip().lower()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        return None
    return sum(float(number) * _UNIT_SECONDS[unit] for number, unit in parts)


def _reset_seconds(value: str, now: float) -> float | None:
    seconds = parse_duration(value)
    if seconds is None:
        return None
    if seconds > 1e12:  # Unix timestamp in milliseconds
        return max(seconds / 1000 - now, 0.0)
    if seconds > 1e9:  # Unix timestamp in seconds
        return max(seconds - now, 0.0)
    return seconds


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_rate_limit(headers: Mapping[str, str], now: float | None = None) -> RateLimitInfo | None:
    now = time.time() if now is None else now
    lower = {k.lower(): v for k, v in headers.items()}
    for prefix, suffix in (
        ("x-ratelimit-", "-requests"),
        ("x-ratelimit-", ""),
        ("ratelimit-", ""),
    ):
        remaining = _int(lower.get(f"{prefix}remaining{suffix}"))
        if remaining is None:
            continue
        reset_raw = lower.get(f"{prefix}reset{suffix}")
        return RateLimitInfo(
            limit=_int(lower.get(f"{prefix}limit{suffix}")),
            remaining=remaining,
            reset_seconds=_reset_seconds(reset_raw, now) if reset_raw else None,
        )
    return None
