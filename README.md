# latamgpt-benchmark

This repository evaluates the [`latam-gpt/CHOCLO`](https://huggingface.co/datasets/latam-gpt/CHOCLO) and [`latam-gpt/Trueque-Benchmark-beta-0.1`](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1) datasets against modern language models and logs the full benchmark to W&B Weave.

This README is intentionally focused on one task: how to add a new model or provider to the benchmark.

## Current providers

- [OpenAI](https://developers.openai.com/api/docs/models)
- [Anthropic](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Google Gemini](https://ai.google.dev/gemini-api/docs/models)
- [Doubleword](https://docs.doubleword.ai/inference-api/models)

## Current environment variables

Required for all runs:

```env
WANDB_API_KEY=...
WEAVE_PROJECT=your-entity/latamgpt-benchmark
HF_TOKEN=...
```

Provider-specific keys:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
DOUBLEWORD_API_KEY=...
```

## Repository layout

- [src/latamgpt_benchmark/datasets.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/datasets.py:1): dataset loading and normalization
- [src/latamgpt_benchmark/models.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/models.py:1): provider clients
- [src/latamgpt_benchmark/evaluator.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/evaluator.py:1): evaluation loop and Weave logging
- [src/latamgpt_benchmark/scoring.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/scoring.py:1): deterministic metrics
- [src/latamgpt_benchmark/judge.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/judge.py:1): optional LLM judge
- [src/latamgpt_benchmark/config.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/config.py:1): default model list and prompts
- [src/latamgpt_benchmark/cli.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/cli.py:1): CLI entrypoint

## Dataset citations

Reference note:

- [docs/dataset-citations-and-comparison.md](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/docs/dataset-citations-and-comparison.md:1)

Trueque:

```bibtex
@software{Trueque_benchmark_beta_0.1,
  title={Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture},
  author={Fuentes, Gonzalo and Arriagada, Alexandra and Henriquez, Clemente and García, M. Alexandra and LatamGPT Team},
  year={2026},
  url={https://huggingface.co/latam-gpt/Trueque-Benchmark-beta-0.1}
}
```

CHOCLO:

```bibtex
@dataset{choclo2026,
  title={CHOCLO: Latin American Cultural Knowledge Benchmark},
  author={Del Solar, Bianca and Carvallo, Andrés and Soto, Álvaro},
  year={2026},
  publisher={Hugging Face},
  note={Developed as part of doctoral research under the supervision of Álvaro Soto}
}
```

## Add a model from an existing provider

If the provider already exists in [models.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/models.py:1), you usually do not need to write new code.

Use one of these paths:

1. Pass the model through the CLI:

```bash
uv run latamgpt-benchmark --model openai:gpt-5.4-mini
uv run latamgpt-benchmark --model doubleword:Qwen/Qwen3.6-35B-A3B-FP8
```

2. Add it to `DEFAULT_MODELS` in [config.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/config.py:1) if it should be part of the default benchmark suite.

The model spec format is always:

```text
provider:model-id
```

## Current curated suites

The current model research is stored in:

- [docs/model-selection-2026-04-22.md](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/docs/model-selection-2026-04-22.md:1)
- [src/latamgpt_benchmark/model_suites.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/model_suites.py:1)
- [docs/dataset-citations-and-comparison.md](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/docs/dataset-citations-and-comparison.md:1)

Available suite names:

- `current-flagships`
- `current-recommended`
- `current-cost-balanced`

Use them like this:

```bash
uv run latamgpt-benchmark --model-suite current-recommended
```

You can also combine a suite with extra explicit models:

```bash
uv run latamgpt-benchmark \
  --model-suite current-cost-balanced \
  --model doubleword:Qwen/Qwen3-14B-FP8
```

## Add a new provider

To add a provider cleanly:

1. Add the provider name to `ModelSpec.parse()` in [config.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/config.py:1).
2. Implement a `BaseModelClient` subclass in [models.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/models.py:1).
3. Register it in `build_model_client()`.
4. Add the required API key to [.env.example](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/.env.example:1).
5. Document one or two recommended model ids in this README.
6. Add or update tests if the parsing or provider selection logic changes.

Provider clients only need to satisfy one contract:

- Input: `system_prompt`, `user_prompt`
- Output: `ModelResponse`

The provider client is responsible for:

- authenticating with the provider
- sending the prompt
- returning normalized text
- returning token usage when available
- returning the raw provider model name, response id, and finish reason when available

Everything else is shared by the benchmark pipeline.

## Doubleword provider

Doubleword is integrated as an OpenAI-compatible provider using the official `openai` Python client with:

- base URL: `https://api.doubleword.ai/v1`
- API key env var: `DOUBLEWORD_API_KEY`

Relevant model ids from the Doubleword catalog for this benchmark:

- `Qwen/Qwen3.6-35B-A3B-FP8`
- `openai/gpt-oss-20b`
- `Qwen/Qwen3-14B-FP8`

Why these were selected:

- `Qwen/Qwen3.6-35B-A3B-FP8`: strong mid-sized reasoning model with a good price/performance profile
- `openai/gpt-oss-20b`: smaller open-weight baseline with low latency
- `Qwen/Qwen3-14B-FP8`: lighter text-only baseline for high-volume runs

## Weave logging contract

Every provider is evaluated through the same loop in [evaluator.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/evaluator.py:1). Each prediction logs:

- dataset row metadata
- model output
- latency
- token usage
- deterministic metrics
- judge metrics when enabled

This means a new provider should not add its own custom evaluation path unless there is a strong reason.

## Contributor checklist

- Keep code and comments in English
- Keep provider integrations simple
- Reuse the shared evaluation loop
- Add models through `provider:model-id`
- Update `.env.example` when introducing a new credential
- Update this README when adding a provider or changing the default benchmark suite

## Validation

Before opening a PR or handing off changes:

```bash
uv run --extra dev ruff check
uv run --extra dev pytest
```
