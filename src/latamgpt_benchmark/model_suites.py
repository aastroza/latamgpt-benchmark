from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSuite:
    name: str
    description: str
    reviewed_on: str
    models: tuple[str, ...]


MODEL_SUITES = {
    "current-flagships": ModelSuite(
        name="current-flagships",
        description="One flagship model per provider for a highest-quality comparison.",
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-5.4",
            "anthropic:claude-opus-4-7",
            "gemini:gemini-2.5-pro",
            "doubleword:Qwen/Qwen3.5-397B-A17B-FP8",
        ),
    ),
    "current-recommended": ModelSuite(
        name="current-recommended",
        description=(
            "Curated benchmark suite with flagship, balanced, and lower-cost variants where "
            "that distinction is clear in the provider documentation."
        ),
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-5.4",
            "openai:gpt-5.4-mini",
            "openai:gpt-5.4-nano",
            "anthropic:claude-opus-4-7",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-haiku-4-5",
            "gemini:gemini-2.5-pro",
            "gemini:gemini-2.5-flash",
            "gemini:gemini-2.5-flash-lite",
            "doubleword:Qwen/Qwen3.5-397B-A17B-FP8",
            "doubleword:Qwen/Qwen3.6-35B-A3B-FP8",
            "doubleword:google/gemma-4-31B-it",
            "doubleword:openai/gpt-oss-20b",
        ),
    ),
    "current-cost-balanced": ModelSuite(
        name="current-cost-balanced",
        description="One practical cost-performance model per provider for larger benchmark runs.",
        reviewed_on="2026-04-22",
        models=(
            "openai:gpt-5.4-mini",
            "anthropic:claude-sonnet-4-6",
            "gemini:gemini-2.5-flash",
            "doubleword:Qwen/Qwen3.6-35B-A3B-FP8",
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
