# Model selection review

Last reviewed: 2026-04-22

This document records the current model suite chosen for the LATAMGPT benchmark and the official sources used to justify it.

## Selection principle

The benchmark datasets are general text QA tasks in Spanish and Latin American context. The most useful benchmark set is not just "the single most expensive model" from each provider. It should include:

- one flagship model
- one balanced price-performance model
- one cheaper or higher-throughput model when the provider clearly offers that tier

That gives us a better view of the provider frontier, the likely practical default, and the lower-cost baseline.

## OpenAI

Official source:

- [OpenAI latest model guide](https://developers.openai.com/api/docs/guides/latest-model)
- [OpenAI models overview](https://developers.openai.com/api/docs/models)

Relevant guidance from OpenAI:

- `gpt-5.4` is the default model for most important work
- `gpt-5.4-mini` is the smaller fast variant
- `gpt-5.4-nano` is the cheapest high-throughput variant

Selected models:

- `openai:gpt-5.4`
- `openai:gpt-5.4-mini`
- `openai:gpt-5.4-nano`

Why:

- This covers the current flagship plus the two clearly documented smaller tiers.
- `gpt-5.4-pro` is intentionally excluded from the main suite because it is positioned for unusually hard tasks and would distort benchmark cost materially on large runs.

## Anthropic

Official source:

- [Anthropic models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Anthropic multilingual support](https://platform.claude.com/docs/en/build-with-claude/multilingual-support)

Relevant guidance from Anthropic:

- `claude-opus-4-7` is the most capable generally available model
- `claude-sonnet-4-6` is the best speed/intelligence combination
- `claude-haiku-4-5` is the fastest model
- Anthropic reports strong multilingual performance, including Spanish close to English-relative performance

Selected models:

- `anthropic:claude-opus-4-7`
- `anthropic:claude-sonnet-4-6`
- `anthropic:claude-haiku-4-5`

Why:

- Anthropic already exposes a clean flagship / balanced / fastest split in the official table.

## Google Gemini

Official source:

- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog)

Relevant guidance from Google:

- `gemini-2.5-pro` is the most advanced model for complex tasks
- `gemini-2.5-flash` is the best price-performance model for low-latency reasoning tasks
- `gemini-2.5-flash-lite` is the fastest and most budget-friendly model in the 2.5 family

Selected models:

- `gemini:gemini-2.5-pro`
- `gemini:gemini-2.5-flash`
- `gemini:gemini-2.5-flash-lite`

Why:

- This gives the clearest current Gemini ladder for benchmarking quality versus efficiency.

## Doubleword

Official source:

- [Doubleword model catalog](https://docs.doubleword.ai/inference-api/models)
- [Doubleword models and pricing](https://docs.doubleword.ai/inference-api/model-pricing)

Selected models:

- `doubleword:Qwen/Qwen3.5-397B-A17B`
- `doubleword:Qwen/Qwen3.6-35B-A3B-FP8`
- `doubleword:google/gemma-4-31B-it`
- `doubleword:openai/gpt-oss-20b`

Why:

- `Qwen/Qwen3.5-397B-A17B` is described by Doubleword as Qwen's most powerful model and is the best open-weight flagship currently documented there.
- `Qwen/Qwen3.6-35B-A3B-FP8` is a strong mid-sized reasoning model with a compelling price/performance profile.
- `google/gemma-4-31B-it` is a strong multilingual open model and explicitly supports 140+ languages.
- `openai/gpt-oss-20b` is a useful low-latency open-weight baseline.

## Saved suites in code

These selections are encoded in:

- [src/latamgpt_benchmark/model_suites.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/model_suites.py:1)

Available suites:

- `current-flagships`
- `current-recommended`
- `current-cost-balanced`

## Example commands

Run the main current suite later:

```bash
uv run latamgpt-benchmark --model-suite current-recommended
```

Run only the flagship comparison:

```bash
uv run latamgpt-benchmark --model-suite current-flagships
```

Run the practical lower-cost suite:

```bash
uv run latamgpt-benchmark --model-suite current-cost-balanced
```
