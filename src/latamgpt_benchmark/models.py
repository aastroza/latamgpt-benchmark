from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from time import perf_counter

from anthropic import Anthropic
from autobatcher import BatchOpenAI
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from latamgpt_benchmark.config import ModelSpec


@dataclass(frozen=True)
class ModelResponse:
    text: str
    latency_seconds: float
    usage: dict[str, int | None]
    raw_model_name: str | None
    response_id: str | None
    finish_reason: str | None


class BaseModelClient:
    def __init__(
        self,
        spec: ModelSpec,
        max_output_tokens: int,
        temperature: float,
    ) -> None:
        self.spec = spec
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        raise NotImplementedError


class OpenAICompatibleModelClient(BaseModelClient):
    def __init__(
        self,
        spec: ModelSpec,
        max_output_tokens: int,
        temperature: float,
        api_key_env_var: str,
        base_url: str | None = None,
    ) -> None:
        super().__init__(spec, max_output_tokens, temperature)
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"{api_key_env_var} is required to use {spec.provider} models.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.spec.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_seconds = perf_counter() - started
        message = response.choices[0].message
        usage = response.usage
        return ModelResponse(
            text=(message.content or "").strip(),
            latency_seconds=latency_seconds,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            raw_model_name=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )


class OpenAIModelClient(OpenAICompatibleModelClient):
    def __init__(self, spec: ModelSpec, max_output_tokens: int, temperature: float) -> None:
        super().__init__(
            spec,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            api_key_env_var="OPENAI_API_KEY",
        )


class DoublewordModelClient(OpenAICompatibleModelClient):
    def __init__(self, spec: ModelSpec, max_output_tokens: int, temperature: float) -> None:
        BaseModelClient.__init__(self, spec, max_output_tokens, temperature)
        api_key = os.getenv("DOUBLEWORD_API_KEY")
        if not api_key:
            raise ValueError("DOUBLEWORD_API_KEY is required to use doubleword models.")
        self.client = BatchOpenAI(
            api_key=api_key,
            base_url="https://api.doubleword.ai/v1",
            batch_window_seconds=1.0,
            poll_interval_seconds=5.0,
            completion_window="1h",
        )

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        started = perf_counter()
        response = asyncio.run(
            self.client.chat.completions.create(
                model=self.spec.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        )
        latency_seconds = perf_counter() - started
        message = response.choices[0].message
        usage = response.usage
        return ModelResponse(
            text=(message.content or "").strip(),
            latency_seconds=latency_seconds,
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            raw_model_name=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )


class AnthropicModelClient(BaseModelClient):
    def __init__(self, spec: ModelSpec, max_output_tokens: int, temperature: float) -> None:
        super().__init__(spec, max_output_tokens, temperature)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required to use Anthropic models.")
        self.client = Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        started = perf_counter()
        request = {
            "model": self.spec.model,
            "system": system_prompt,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.temperature != 0.0:
            request["temperature"] = self.temperature
        response = self.client.messages.create(**request)
        latency_seconds = perf_counter() - started
        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        usage = response.usage
        return ModelResponse(
            text="".join(text_blocks).strip(),
            latency_seconds=latency_seconds,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": _sum_optional(
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                ),
            },
            raw_model_name=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            finish_reason=getattr(response, "stop_reason", None),
        )


class GeminiModelClient(BaseModelClient):
    def __init__(self, spec: ModelSpec, max_output_tokens: int, temperature: float) -> None:
        super().__init__(spec, max_output_tokens, temperature)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to use Gemini models.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        started = perf_counter()
        response = self.client.models.generate_content(
            model=self.spec.model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        latency_seconds = perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        return ModelResponse(
            text=(response.text or "").strip(),
            latency_seconds=latency_seconds,
            usage={
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            },
            raw_model_name=getattr(response, "model_version", None),
            response_id=getattr(response, "response_id", None),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )


def build_model_client(
    spec: ModelSpec,
    max_output_tokens: int,
    temperature: float,
) -> BaseModelClient:
    if spec.provider == "openai":
        return OpenAIModelClient(spec, max_output_tokens=max_output_tokens, temperature=temperature)
    if spec.provider == "anthropic":
        return AnthropicModelClient(spec, max_output_tokens=max_output_tokens, temperature=temperature)
    if spec.provider == "gemini":
        return GeminiModelClient(spec, max_output_tokens=max_output_tokens, temperature=temperature)
    if spec.provider == "doubleword":
        return DoublewordModelClient(spec, max_output_tokens=max_output_tokens, temperature=temperature)
    raise ValueError(f"Unsupported provider '{spec.provider}'.")


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left + right
