"""Turn provider error messages and HTTP statuses into JARVIS errors."""

from __future__ import annotations

from jarvis.core.errors import (
    NotConfiguredError,
    PaymentRequiredError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)

# Billing words are checked first and map to a permanent block: when in doubt
# about money, stop using the agent (the safe direction).
_BILLING = ("payment", "billing", "insufficient", "credit", "balance", "402", "top up")
_RATE = ("rate limit", "rate_limit", "ratelimit", "429", "too many requests", "quota")
_ACCESS = ("free tier can only be used", "unauthorized", "forbidden", "invalid api key", "401")


def error_from_message(message: str, retry_after: float | None = None) -> ProviderError:
    text = message.lower()
    if any(word in text for word in _BILLING):
        return PaymentRequiredError(message)
    if any(word in text for word in _RATE):
        return RateLimitError(message, retry_after=retry_after)
    if any(word in text for word in _ACCESS):
        return NotConfiguredError(message)
    return ProviderUnavailableError(message, retry_after=retry_after)


def error_from_status(status: int, message: str, retry_after: float | None = None) -> ProviderError:
    if status == 402:
        return PaymentRequiredError(f"HTTP 402: {message}")
    if status == 429:
        return RateLimitError(f"HTTP 429: {message}", retry_after=retry_after)
    if status in (401, 403):
        return NotConfiguredError(f"HTTP {status}: {message}")
    if status >= 500:
        return ProviderUnavailableError(f"HTTP {status}: {message}", retry_after=retry_after)
    return error_from_message(f"HTTP {status}: {message}", retry_after)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
