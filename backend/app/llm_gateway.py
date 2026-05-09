# =============================================================================
# MEGA AI — LLM Gateway
# =============================================================================
# THE ONLY PLACE LLM CALLS HAPPEN IN THE ENTIRE CODEBASE.
# This is the "swappable engine" of the V16 chassis.
# Model is configured via .env ONLY — never in code.
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when provider returns rate limit response."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when provider times out."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when API key is invalid."""
    pass


# =============================================================================
# Response Objects
# =============================================================================

@dataclass
class LLMResponse:
    """Standardized response from ANY LLM provider."""

    content: str
    model: str
    provider: str
    latency_ms: int
    tokens_used: int
    tokens_remaining: Optional[int] = None
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""

    provider: str
    model: str
    prompt_hash: str
    response_hash: str
    latency_ms: int
    tokens_used: int
    success: bool
    error: Optional[str] = None
    tier: str = "unknown"


# =============================================================================
# Provider Protocol
# =============================================================================

class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> "LLMResponse":
        ...

    @property
    def name(self) -> str:
        ...


# =============================================================================
# Gemini Provider
# =============================================================================

class GeminiProvider:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ) -> None:

        self.api_key = api_key
        self.model = model
        self._display_name = "Google Gemini"
        self._session = None

    @property
    def name(self) -> str:
        return self._display_name

    @retry(
        retry=retry_if_exception_type(
            (
                LLMTimeoutError,
                LLMRateLimitError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=2,
            max=30,
        ),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:

        start_time = time.time()

        try:

            import google.generativeai as genai

            genai.configure(
                api_key=self.api_key
            )

            model = genai.GenerativeModel(
                self.model
            )

            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                ),
                timeout=30.0,
            )

            latency_ms = int(
                (time.time() - start_time) * 1000
            )

            content = (
                response.text
                if hasattr(response, "text")
                else str(response)
            )

            tokens_used = (
                len(prompt.split())
                + len(content.split())
            )

            return LLMResponse(
                content=content,
                model=self.model,
                provider="gemini",
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )

        except asyncio.TimeoutError:

            raise LLMTimeoutError(
                "Gemini request timed out after 30s"
            )

        except Exception as e:

            error_str = str(e).lower()

            if (
                "rate limit" in error_str
                or "429" in error_str
            ):

                raise LLMRateLimitError(
                    f"Gemini rate limit: {e}"
                )

            elif (
                "invalid api key" in error_str
                or "401" in error_str
            ):

                raise LLMAuthenticationError(
                    f"Gemini auth failed: {e}"
                )

            raise LLMProviderError(
                f"Gemini error: {e}"
            )


# =============================================================================
# Groq Provider
# =============================================================================

class GroqProvider:

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:

        self.api_key = api_key
        self.model = model
        self._display_name = "Groq"

        self._base_url = (
            "https://api.groq.com/openai/v1/chat/completions"
        )

    @property
    def name(self) -> str:
        return self._display_name

    @retry(
        retry=retry_if_exception_type(
            (
                LLMTimeoutError,
                LLMRateLimitError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=20,
        ),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:

        import aiohttp

        start_time = time.time()

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:

            async with aiohttp.ClientSession() as session:

                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=30
                    ),
                ) as response:

                    if response.status == 429:

                        raise LLMRateLimitError(
                            "Groq rate limit exceeded"
                        )

                    elif response.status == 401:

                        raise LLMAuthenticationError(
                            "Groq API key invalid"
                        )

                    response.raise_for_status()

                    data = await response.json()

                    latency_ms = int(
                        (
                            time.time()
                            - start_time
                        ) * 1000
                    )

                    content = (
                        data["choices"][0]
                        ["message"]["content"]
                    )

                    usage = data.get(
                        "usage",
                        {},
                    )

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="groq",
                        latency_ms=latency_ms,
                        tokens_used=usage.get(
                            "total_tokens",
                            len(prompt.split())
                            + len(content.split()),
                        ),
                    )

        except asyncio.TimeoutError:

            raise LLMTimeoutError(
                "Groq request timed out after 30s"
            )

        except LLMProviderError:
            raise

        except Exception as e:

            raise LLMProviderError(
                f"Groq error: {e}"
            )


# =============================================================================
# DeepSeek Provider
# =============================================================================

class DeepSeekProvider:

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-pro",
    ) -> None:

        self.api_key = api_key
        self.model = model
        self._display_name = "DeepSeek"

        self._base_url = (
            "https://api.deepseek.com/v1/chat/completions"
        )

    @property
    def name(self) -> str:
        return self._display_name

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:

        import aiohttp

        start_time = time.time()

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:

            async with aiohttp.ClientSession() as session:

                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=45
                    ),
                ) as response:

                    response.raise_for_status()

                    data = await response.json()

                    latency_ms = int(
                        (
                            time.time()
                            - start_time
                        ) * 1000
                    )

                    content = (
                        data["choices"][0]
                        ["message"]["content"]
                    )

                    usage = data.get(
                        "usage",
                        {},
                    )

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="deepseek",
                        latency_ms=latency_ms,
                        tokens_used=usage.get(
                            "total_tokens",
                            0,
                        ),
                    )

        except asyncio.TimeoutError:

            raise LLMTimeoutError(
                "DeepSeek request timed out after 45s"
            )

        except Exception as e:

            raise LLMProviderError(
                f"DeepSeek error: {e}"
            )


# =============================================================================
# Anthropic Provider
# =============================================================================

class AnthropicProvider:

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
    ) -> None:

        self.api_key = api_key
        self.model = model
        self._display_name = "Anthropic"

    @property
    def name(self) -> str:
        return self._display_name

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:

        import aiohttp

        start_time = time.time()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        try:

            async with aiohttp.ClientSession() as session:

                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=60
                    ),
                ) as response:

                    response.raise_for_status()

                    data = await response.json()

                    latency_ms = int(
                        (
                            time.time()
                            - start_time
                        ) * 1000
                    )

                    content = (
                        data["content"][0]["text"]
                    )

                    usage = data.get(
                        "usage",
                        {},
                    )

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        provider="anthropic",
                        latency_ms=latency_ms,
                        tokens_used=(
                            usage.get(
                                "input_tokens",
                                0,
                            )
                            + usage.get(
                                "output_tokens",
                                0,
                            )
                        ),
                    )

        except asyncio.TimeoutError:

            raise LLMTimeoutError(
                "Anthropic request timed out after 60s"
            )

        except Exception as e:

            raise LLMProviderError(
                f"Anthropic error: {e}"
            )


# =============================================================================
# LLM Gateway
# =============================================================================

class LLMGateway:
    """
    Central gateway for all LLM calls.
    """

    def __init__(
        self,
        config: Optional[
            Dict[str, str]
        ] = None,
    ) -> None:

        self.call_history: List[
            LLMCallRecord
        ] = []

        self._providers: List[
            LLMProvider
        ] = []

        self._provider_configs: List[
            Dict[str, str]
        ] = []

        if config is None:

            from app.config import (
                get_llm_config_with_fallback,
            )

            self._provider_configs = (
                get_llm_config_with_fallback()
            )

        else:

            self._provider_configs = [
                config
            ]

        self._init_providers()

    def _init_providers(self) -> None:
        """
        Initialize providers.
        """

        for cfg in self._provider_configs:

            provider_name = cfg.get(
                "provider",
                "",
            ).lower()

            api_key = cfg.get(
                "api_key",
                "",
            )

            model = cfg.get(
                "model",
                "",
            )

            if not api_key:

                logger.warning(
                    f"No API key for "
                    f"{provider_name}, skipping"
                )

                continue

            try:

                if provider_name == "gemini":

                    provider = GeminiProvider(
                        api_key=api_key,
                        model=model,
                    )

                elif provider_name == "groq":

                    provider = GroqProvider(
                        api_key=api_key,
                        model=model,
                    )

                elif provider_name == "deepseek":

                    provider = DeepSeekProvider(
                        api_key=api_key,
                        model=model,
                    )

                elif provider_name == "anthropic":

                    provider = AnthropicProvider(
                        api_key=api_key,
                        model=model,
                    )

                else:

                    logger.warning(
                        f"Unknown provider: "
                        f"{provider_name}"
                    )

                    continue

                self._providers.append(
                    provider
                )

                logger.info(
                    f"Initialized "
                    f"{provider.name} "
                    f"with model {model}"
                )

            except Exception as e:

                logger.error(
                    f"Failed to initialize "
                    f"{provider_name}: {e}"
                )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        job_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate text with automatic fallback.
        """

        if not self._providers:

            raise LLMProviderError(
                "No LLM providers configured. "
                "Check API keys in .env"
            )

        prompt_hash = hashlib.sha256(
            prompt.encode()
        ).hexdigest()[:16]

        last_error: Optional[
            Exception
        ] = None

        for idx, provider in enumerate(
            self._providers
        ):

            try:

                logger.info(
                    f"LLM call attempt "
                    f"{idx + 1}/"
                    f"{len(self._providers)}: "
                    f"{provider.name}",
                    extra={
                        "job_id": job_id,
                        "agent_id": agent_id,
                        "provider": provider.name,
                        "prompt_hash": prompt_hash,
                    },
                )

                response = await provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

                record = LLMCallRecord(
                    provider=provider.name,
                    model=response.model,
                    prompt_hash=prompt_hash,
                    response_hash=hashlib.sha256(
                        response.content.encode()
                    ).hexdigest()[:16],
                    latency_ms=response.latency_ms,
                    tokens_used=response.tokens_used,
                    success=True,
                    tier=(
                        [
                            "primary",
                            "fallback",
                            "tertiary",
                        ][idx]
                        if idx < 3
                        else "unknown"
                    ),
                )

                self.call_history.append(
                    record
                )

                logger.info(
                    f"LLM call successful: "
                    f"{provider.name} "
                    f"in {response.latency_ms}ms",
                    extra={
                        "job_id": job_id,
                        "agent_id": agent_id,
                        "tokens_used": (
                            response.tokens_used
                        ),
                    },
                )

                return response

            except LLMProviderError as e:

                last_error = e

                logger.warning(
                    f"Provider {provider.name} "
                    f"failed: {e}. "
                    f"Trying next...",
                    extra={
                        "job_id": job_id,
                        "agent_id": agent_id,
                    },
                )

                record = LLMCallRecord(
                    provider=provider.name,
                    model=getattr(
                        provider,
                        "model",
                        "unknown",
                    ),
                    prompt_hash=prompt_hash,
                    response_hash="",
                    latency_ms=0,
                    tokens_used=0,
                    success=False,
                    error=str(e),
                    tier=(
                        [
                            "primary",
                            "fallback",
                            "tertiary",
                        ][idx]
                        if idx < 3
                        else "unknown"
                    ),
                )

                self.call_history.append(
                    record
                )

                continue

        raise LLMProviderError(
            f"All {len(self._providers)} "
            f"providers failed. "
            f"Last error: {last_error}"
        )

    def get_call_stats(
        self,
    ) -> Dict[str, Any]:
        """
        Return call statistics.
        """

        total = len(
            self.call_history
        )

        successful = sum(
            1
            for r in self.call_history
            if r.success
        )

        failed = total - successful

        latencies = [
            r.latency_ms
            for r in self.call_history
            if r.success
        ]

        avg_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else 0
        )

        provider_breakdown: Dict[
            str,
            Dict[str, int],
        ] = {}

        for record in self.call_history:

            name = record.provider

            if name not in provider_breakdown:

                provider_breakdown[name] = {
                    "calls": 0,
                    "success": 0,
                    "failed": 0,
                }

            provider_breakdown[name][
                "calls"
            ] += 1

            if record.success:

                provider_breakdown[name][
                    "success"
                ] += 1

            else:

                provider_breakdown[name][
                    "failed"
                ] += 1

        return {
            "total_calls": total,
            "successful_calls": successful,
            "failed_calls": failed,
            "avg_latency_ms": round(
                avg_latency,
                2,
            ),
            "provider_breakdown": (
                provider_breakdown
            ),
        }

    def clear_history(self) -> None:
        """
        Clear call history.
        """

        self.call_history.clear()