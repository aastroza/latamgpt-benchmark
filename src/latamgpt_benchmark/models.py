from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

from openai import OpenAI

from latamgpt_benchmark.config import ModelSpec


TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled", "cancelling"}


@dataclass(frozen=True)
class BatchInfo:
    batch_id: str
    input_file_id: str
    status: str
    output_file_id: str | None
    error_file_id: str | None
    request_counts: dict[str, int | None]


@dataclass(frozen=True)
class DownloadedBatchFile:
    text: str
    headers: dict[str, str]


class BaseBatchClient:
    def __init__(self, spec: ModelSpec, api_key_env_var: str, base_url: str | None = None) -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"{api_key_env_var} is required to use {spec.provider} models.")
        self.spec = spec
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def build_request_body(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "model": self.spec.model,
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    def upload_batch_file(self, path: str) -> str:
        with open(path, "rb") as handle:
            uploaded = self.client.files.create(file=handle, purpose="batch")
        return uploaded.id

    def create_batch(
        self,
        input_file_id: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> BatchInfo:
        self.validate_completion_window(completion_window)
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )
        return _batch_info_from_sdk(batch)

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        return _batch_info_from_sdk(self.client.batches.retrieve(batch_id))

    def download_output_file(self, file_id: str) -> DownloadedBatchFile:
        output_url = f"{self.base_url}/files/{file_id}/content"
        http_request = request.Request(
            output_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with request.urlopen(http_request) as response:
            payload = response.read().decode("utf-8")
            headers = {key: value for key, value in response.headers.items()}
        return DownloadedBatchFile(text=payload, headers=headers)

    def validate_completion_window(self, completion_window: str) -> None:
        if completion_window != "24h":
            raise ValueError(
                f"{self.spec.provider} batches only support completion_window='24h'."
            )


class OpenAIBatchClient(BaseBatchClient):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, api_key_env_var="OPENAI_API_KEY")


class DoublewordBatchClient(BaseBatchClient):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(
            spec,
            api_key_env_var="DOUBLEWORD_API_KEY",
            base_url="https://api.doubleword.ai/v1",
        )

    def validate_completion_window(self, completion_window: str) -> None:
        if completion_window not in {"24h", "1h"}:
            raise ValueError("doubleword batches support completion_window values '24h' or '1h'.")


def build_batch_client(spec: ModelSpec) -> BaseBatchClient:
    if spec.provider == "openai":
        return OpenAIBatchClient(spec)
    if spec.provider == "doubleword":
        return DoublewordBatchClient(spec)
    raise ValueError(f"Unsupported provider '{spec.provider}'.")


def parse_batch_output_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("error") is not None:
        raise ValueError(f"Batch request {row.get('custom_id')} failed: {row['error']}")
    response = row.get("response") or {}
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(
            f"Batch request {row.get('custom_id')} returned unexpected status code {status_code}."
        )
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"Batch request {row.get('custom_id')} did not include any choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text = "".join(_content_part_to_text(part) for part in content).strip()
    else:
        text = str(content).strip()
    usage = body.get("usage") or {}
    return {
        "text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "raw_model_name": body.get("model"),
        "response_id": body.get("id"),
        "finish_reason": choices[0].get("finish_reason"),
    }


def _batch_info_from_sdk(batch: Any) -> BatchInfo:
    request_counts = getattr(batch, "request_counts", None)
    return BatchInfo(
        batch_id=getattr(batch, "id"),
        input_file_id=getattr(batch, "input_file_id"),
        status=getattr(batch, "status"),
        output_file_id=getattr(batch, "output_file_id", None),
        error_file_id=getattr(batch, "error_file_id", None),
        request_counts={
            "total": getattr(request_counts, "total", None),
            "completed": getattr(request_counts, "completed", None),
            "failed": getattr(request_counts, "failed", None),
        },
    )


def _content_part_to_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if part.get("type") == "text":
            return str(part.get("text", ""))
        return json.dumps(part, ensure_ascii=False)
    return str(part)
