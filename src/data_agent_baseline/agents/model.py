from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import APIError, APITimeoutError, AzureOpenAI, OpenAI

REQUEST_TIMEOUT = 80  # seconds per API call

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking_tokens(content: str) -> str:
    """Remove <think>...</think> blocks that vLLM reasoning parsers may include."""
    return _THINK_RE.sub("", content).strip()


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            from httpx import Timeout

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "timeout": Timeout(REQUEST_TIMEOUT, connect=10.0),
            }
            if "services.ai.azure.com" in self.api_base:
                kwargs["default_headers"] = {"api-key": self.api_key}
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, messages: list[ModelMessage], *, thinking: bool = True) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_completion_tokens": 4096,
        }
        if self.temperature is not None and "gpt-5" not in self.model.lower():
            kwargs["temperature"] = self.temperature
        if "qwen" in self.model.lower():
            if "deepinfra" in self.api_base:
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            else:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

        try:
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(client.chat.completions.create, **kwargs)
            try:
                response = future.result(timeout=REQUEST_TIMEOUT)
            except FuturesTimeout:
                pool.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"Model API error: wall-clock timeout after {REQUEST_TIMEOUT}s")
            finally:
                pool.shutdown(wait=False)
        except (APITimeoutError, APIError) as exc:
            raise RuntimeError(f"Model API error: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return _strip_thinking_tokens(content)


def _normalize_azure_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint must not be empty.")
    if "/openai/" in endpoint:
        parsed = urlparse(endpoint)
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint


class AzureOpenAIModelAdapter:
    def __init__(
        self,
        *,
        deployment_name: str,
        api_key: str,
        endpoint: str,
        api_version: str,
        temperature: float,
    ) -> None:
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = _normalize_azure_endpoint(endpoint)
        self.api_version = api_version
        self.temperature = temperature

    def complete(self, messages: list[ModelMessage], **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError("Missing Azure OpenAI API key in config.agent.api_key.")

        from httpx import Timeout

        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            timeout=Timeout(REQUEST_TIMEOUT, connect=10.0),
        )

        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                temperature=self.temperature,
                timeout=REQUEST_TIMEOUT,
            )
        except (APITimeoutError, APIError) as exc:
            raise RuntimeError(f"Azure OpenAI request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Azure OpenAI response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Azure OpenAI response missing text content.")
        return content


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
