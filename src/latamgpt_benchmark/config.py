from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "Answer in Spanish with a focus on factual accuracy. "
    "If you do not know the answer, say so explicitly and do not invent details. "
    "Keep the answer brief and useful."
)


DEFAULT_JUDGE_PROMPT = """
You are a strict evaluator for model answers on Latin American knowledge benchmarks.

Compare:
- the original question
- the reference answer
- the model answer

Score from 0.0 to 1.0:
- correctness_score: factual accuracy with respect to the reference answer
- completeness_score: coverage of the important points
- uncertainty_handling_score: whether uncertainty is handled appropriately when needed
- overall_score: overall quality

Also return:
- verdict: one of ["correct", "partially_correct", "incorrect"]
- justification: a brief explanation in English

Return only valid JSON using this schema:
{
  "correctness_score": 0.0,
  "completeness_score": 0.0,
  "uncertainty_handling_score": 0.0,
  "overall_score": 0.0,
  "verdict": "correct",
  "justification": "..."
}
""".strip()


DEFAULT_MODELS = [
    "openai:gpt-4.1-mini",
    "openai:gpt-5-mini",
    "openai:gpt-5-nano",
    "doubleword:Qwen/Qwen3.5-4B",
    "doubleword:google/gemma-4-31B-it",
    "doubleword:openai/gpt-oss-20b",
]

DEFAULT_JUDGE_MODEL = "openai:gpt-4.1-mini"


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def slug(self) -> str:
        return self.name.replace(":", "__").replace("/", "_")

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        provider, separator, model = value.partition(":")
        if not separator or not provider or not model:
            raise ValueError(
                f"Invalid model spec '{value}'. Expected format 'provider:model-id'."
            )
        normalized_provider = provider.strip().lower()
        if normalized_provider not in {"openai", "doubleword"}:
            raise ValueError(
                f"Unsupported provider '{normalized_provider}'. Use openai or doubleword."
            )
        return cls(provider=normalized_provider, model=model.strip())


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: list[str]
    models: list[ModelSpec]
    judge_model: ModelSpec | None
    weave_project: str | None
    output_dir: Path
    run_name: str
    system_prompt: str
    judge_prompt: str
    split: str = "train"
    max_samples: int | None = None
    shuffle: bool = False
    seed: int = 7
    num_shards: int = 1
    shard_index: int = 0
    max_output_tokens: int = 256
    judge_max_output_tokens: int = 256
    temperature: float = 0.0
    answer_completion_window: str = "24h"
    judge_completion_window: str = "24h"
    max_requests_per_batch: int = 5000

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["models"] = [model.name for model in self.models]
        payload["judge_model"] = self.judge_model.name if self.judge_model else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            datasets=list(payload["datasets"]),
            models=[ModelSpec.parse(value) for value in payload["models"]],
            judge_model=(
                ModelSpec.parse(payload["judge_model"]) if payload.get("judge_model") else None
            ),
            weave_project=payload.get("weave_project"),
            output_dir=Path(payload["output_dir"]),
            run_name=payload["run_name"],
            system_prompt=payload["system_prompt"],
            judge_prompt=payload["judge_prompt"],
            split=payload.get("split", "train"),
            max_samples=payload.get("max_samples"),
            shuffle=payload.get("shuffle", False),
            seed=payload.get("seed", 7),
            num_shards=payload.get("num_shards", 1),
            shard_index=payload.get("shard_index", 0),
            max_output_tokens=payload.get("max_output_tokens", 256),
            judge_max_output_tokens=payload.get("judge_max_output_tokens", 256),
            temperature=payload.get("temperature", 0.0),
            answer_completion_window=payload.get("answer_completion_window", "24h"),
            judge_completion_window=payload.get("judge_completion_window", "24h"),
            max_requests_per_batch=payload.get("max_requests_per_batch", 5000),
        )
