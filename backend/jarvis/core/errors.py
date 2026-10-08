"""Errors raised by agents. The agent manager maps each one to a health change."""

from __future__ import annotations


class JarvisError(Exception):
    """Base class for every JARVIS error."""


class ProviderError(JarvisError):
    """An agent failed to answer. Subclasses tell the manager how to react."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """Free quota or rate limit reached → cooldown, then try again."""


class ProviderUnavailableError(ProviderError):
    """Network error, timeout, 5xx, provider down → exponential backoff."""


class InvalidResponseError(ProviderError):
    """The provider answered with something we could not use → backoff."""


class NotConfiguredError(ProviderError):
    """Missing API key, missing binary, or wrong credentials → unconfigured."""


class PaymentRequiredError(ProviderError):
    """HTTP 402 or an explicit billing error → blocked for good."""


class CostViolationError(ProviderError):
    """The cost guard refused the agent or saw a non-zero cost → blocked for good."""


class AllAgentsFailedError(JarvisError):
    """No agent could answer this request."""

    def __init__(self, message: str, attempts: list | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []
