"""Generic adapter for OpenAI-compatible chat APIs (Groq, OpenRouter, Mistral…).

Adding one of these providers is configuration, not code. Free-only safety:
- the model name must match the provider's free pattern (e.g. ``:free`` on
  OpenRouter), otherwise the agent is blocked before any call;
- HTTP 402 and billing messages block the agent for good;
- a reported ``usage.cost`` above zero is caught by the cost guard.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from jarvis.agents.errors import error_from_status, parse_retry_after
from jarvis.core.errors import (
    CostViolationError,
    InvalidResponseError,
    NotConfiguredError,
    ProviderError,
    ProviderUnavailableError,
)
from jarvis.core.provider import AIProvider, HealthReport
from jarvis.core.types import AgentInfo, AIRequest, AIResponse, Health


class OpenAICompatAgent(AIProvider):
    def __init__(
        self,
        info: AgentInfo,
        *,
        base_url: str,
        api_key_env: str,
        free_model_pattern: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.info = info
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.free_model_pattern = free_model_pattern
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    # The key is read on demand so it never sits in a config object or a log.
    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise NotConfiguredError(f"set {self.api_key_env} in .env to enable {self.info.name}")
        return key

    def _check_free_model(self) -> None:
        if self.free_model_pattern and not re.search(self.free_model_pattern, self.info.model):
            raise CostViolationError(
                f"model {self.info.model!r} does not match the free pattern "
                f"{self.free_model_pattern!r}"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key()}"}

    async def check_health(self) -> HealthReport:
        try:
            self._check_free_model()
            headers = self._headers()
        except CostViolationError as exc:
            return HealthReport(Health.BLOCKED, str(exc))
        except NotConfiguredError as exc:
            return HealthReport(Health.UNCONFIGURED, str(exc))
        try:
            # Listing models does not spend generation quota.
            response = await self._client.get(f"{self.base_url}/models", headers=headers)
        except httpx.HTTPError as exc:
            return HealthReport(Health.OFFLINE, f"cannot reach {self.base_url}: {exc}")
        if response.status_code in (401, 403):
            return HealthReport(Health.UNCONFIGURED, f"HTTP {response.status_code}: check the key")
        if response.status_code >= 400:
            return HealthReport(Health.OFFLINE, f"HTTP {response.status_code} on /models")
        try:
            ids = {m.get("id") for m in response.json().get("data", [])}
        except ValueError:
            return HealthReport(Health.OFFLINE, "invalid /models response")
        if ids and self.info.model not in ids:
            return HealthReport(Health.OFFLINE, f"model {self.info.model} is not offered anymore")
        return HealthReport(Health.AVAILABLE)

    async def generate(self, request: AIRequest) -> AIResponse:
        self._check_free_model()
        started = time.perf_counter()
        payload: dict = {
            "model": self.info.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"{self.info.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"{self.info.name} unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise self._error(response)
        try:
            body = response.json()
            text = (body["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(f"{self.info.name}: unexpected response shape") from exc
        if not text:
            raise InvalidResponseError(f"{self.info.name}: empty answer")
        usage = body.get("usage") or {}
        cost = usage.get("cost")
        return AIResponse(
            text=text,
            agent_id=self.id,
            model=self.info.model,
            cost=float(cost) if cost is not None else None,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        try:
            body = response.json()
            error = body.get("error", body)
            message = error.get("message") if isinstance(error, dict) else str(error)
        except ValueError:
            message = response.text[:300]
        retry = parse_retry_after(response.headers.get("retry-after"))
        return error_from_status(response.status_code, f"{self.info.name}: {message}", retry)

    async def close(self) -> None:
        await self._client.aclose()
