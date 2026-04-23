from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSuite:
    name: str
    description: str
    reviewed_on: str
    models: tuple[str, ...]


MODEL_SUITES = {
    "benchmark-default": ModelSuite(
        name="benchmark-default",
        description="OpenAI conservative set with a flagship plus one Qwen, Gemma, and GPT-OSS.",
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-4.1-mini",
            "openai:gpt-5.4",
            "openai:gpt-5-nano",
            "doubleword:Qwen/Qwen3.5-4B",
            "doubleword:google/gemma-4-31B-it",
            "doubleword:openai/gpt-oss-20b",
        ),
    ),
    "openai-conservative": ModelSuite(
        name="openai-conservative",
        description="Current OpenAI benchmark set without the GPT models already reported by CHOCLO.",
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-4.1-mini",
            "openai:gpt-5.4",
            "openai:gpt-5-nano",
        ),
    ),
    "doubleword-small": ModelSuite(
        name="doubleword-small",
        description="One small Qwen plus Gemma and GPT-OSS for inexpensive async benchmarking.",
        reviewed_on="2026-04-22",
        models=(
            "doubleword:Qwen/Qwen3.5-4B",
            "doubleword:google/gemma-4-31B-it",
            "doubleword:openai/gpt-oss-20b",
        ),
    ),
    "cost-minimal": ModelSuite(
        name="cost-minimal",
        description="Smallest practical pair for cheap smoke runs and pipeline validation.",
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-4.1-mini",
            "doubleword:Qwen/Qwen3.5-4B",
        ),
    ),
}


def resolve_model_list(model_names: list[str], suite_names: list[str]) -> list[str]:
    resolved: list[str] = []
    for suite_name in suite_names:
        suite = MODEL_SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(sorted(MODEL_SUITES))
            raise ValueError(f"Unknown model suite '{suite_name}'. Available suites: {available}.")
        resolved.extend(suite.models)
    resolved.extend(model_names)
    return _deduplicate(resolved)


def available_suite_names() -> list[str]:
    return sorted(MODEL_SUITES)


def _deduplicate(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            deduplicated.append(value)
            seen.add(value)
    return deduplicated
