from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


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
    "openai:gpt-5.4-mini",
    "anthropic:claude-sonnet-4-6",
    "gemini:gemini-2.5-flash",
    "doubleword:Qwen/Qwen3.6-35B-A3B-FP8",
]

DEFAULT_JUDGE_MODEL = "openai:gpt-5.4-mini"


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
        if normalized_provider not in {"openai", "anthropic", "gemini", "doubleword"}:
            raise ValueError(
                "Unsupported provider "
                f"'{normalized_provider}'. Use openai, anthropic, gemini, or doubleword."
            )
        return cls(provider=normalized_provider, model=model.strip())


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: list[str]
    models: list[ModelSpec]
    judge_model: ModelSpec | None
    weave_project: str
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

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["models"] = [model.name for model in self.models]
        payload["judge_model"] = self.judge_model.name if self.judge_model else None
        return payload
