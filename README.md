# latamgpt-benchmark

Evaluates [`latam-gpt/CHOCLO`](https://huggingface.co/datasets/latam-gpt/CHOCLO) and [`latam-gpt/Trueque-Benchmark-beta-0.1`](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1) against modern language models and logs the full benchmark to W&B Weave.

---

## Table of contents

- [Auditing the benchmark](#auditing-the-benchmark)
- [Environment variables](#environment-variables)
- [Running a benchmark](#running-a-benchmark)
- [Model suites](#model-suites)
- [Repository layout](#repository-layout)
- [Adding a model from an existing provider](#adding-a-model-from-an-existing-provider)
- [Adding a new provider](#adding-a-new-provider)
- [Doubleword provider notes](#doubleword-provider-notes)
- [Weave logging contract](#weave-logging-contract)
- [Dataset citations](#dataset-citations)
- [Contributor checklist](#contributor-checklist)

---

## Auditing the benchmark

If you want to inspect or reproduce results, these are the files that matter:

| What you want to audit | Where to look |
|---|---|
| How datasets are loaded and normalized | [`src/latamgpt_benchmark/datasets.py`](src/latamgpt_benchmark/datasets.py) |
| How the evaluation loop works | [`src/latamgpt_benchmark/evaluator.py`](src/latamgpt_benchmark/evaluator.py) |
| Deterministic metrics (exact match, etc.) | [`src/latamgpt_benchmark/scoring.py`](src/latamgpt_benchmark/scoring.py) |
| LLM judge logic and prompts | [`src/latamgpt_benchmark/judge.py`](src/latamgpt_benchmark/judge.py) |
| Default system prompts and model list | [`src/latamgpt_benchmark/config.py`](src/latamgpt_benchmark/config.py) |
| Curated model suites | [`src/latamgpt_benchmark/model_suites.py`](src/latamgpt_benchmark/model_suites.py) |
| Dataset comparison notes | [`docs/dataset-citations-and-comparison.md`](docs/dataset-citations-and-comparison.md) |
| Model selection rationale (April 2026) | [`docs/model-selection-2026-04-22.md`](docs/model-selection-2026-04-22.md) |

Every provider is evaluated through the same shared loop in [`evaluator.py`](src/latamgpt_benchmark/evaluator.py). Each prediction logs dataset row metadata, model output, latency, token usage, deterministic metrics, and judge metrics when enabled. There are no provider-specific evaluation paths.

---

## Environment variables

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

See [`.env.example`](.env.example) for the full template.

---

## Running a benchmark

```bash
# Single model
uv run latamgpt-benchmark --model anthropic:claude-sonnet-4-20250514

# Full suite
uv run latamgpt-benchmark --model-suite current-recommended

# Suite plus extra models
uv run latamgpt-benchmark \
  --model-suite current-cost-balanced \
  --model doubleword:Qwen/Qwen3-14B-FP8
```

---

## Model suites

Curated suites are defined in [`src/latamgpt_benchmark/model_suites.py`](src/latamgpt_benchmark/model_suites.py). Available names:

| Suite | Use case |
|---|---|
| `current-flagships` | Best available models regardless of cost |
| `current-recommended` | Balanced quality and cost |
| `current-cost-balanced` | High-volume or budget runs |

Selection rationale is documented in [`docs/model-selection-2026-04-22.md`](docs/model-selection-2026-04-22.md).

---

## Repository layout

```
src/latamgpt_benchmark/
├── datasets.py      # Dataset loading and normalization
├── models.py        # Provider clients
├── evaluator.py     # Evaluation loop and Weave logging
├── scoring.py       # Deterministic metrics
├── judge.py         # Optional LLM judge
├── config.py        # Default model list and prompts
├── model_suites.py  # Curated model suites
└── cli.py           # CLI entrypoint

docs/
├── dataset-citations-and-comparison.md
└── model-selection-2026-04-22.md
```

---

## Adding a model from an existing provider

If the provider already exists in [`models.py`](src/latamgpt_benchmark/models.py), no new code is needed.

The model spec format is:

```
provider:model-id
```

**Option 1 — pass it through the CLI:**

```bash
uv run latamgpt-benchmark --model openai:gpt-5.4-mini
uv run latamgpt-benchmark --model doubleword:Qwen/Qwen3.6-35B-A3B-FP8
```

**Option 2 — add it to `DEFAULT_MODELS` in [`config.py`](src/latamgpt_benchmark/config.py)** if it should run by default.

Current providers: OpenAI, Anthropic, Google Gemini, Doubleword.

---

## Adding a new provider

Provider clients have a single contract:

- **Input:** `system_prompt`, `user_prompt`
- **Output:** `ModelResponse` (normalized text, token usage, raw model name, response id, finish reason)

Steps:

1. Add the provider name to `ModelSpec.parse()` in [`config.py`](src/latamgpt_benchmark/config.py).
2. Implement a `BaseModelClient` subclass in [`models.py`](src/latamgpt_benchmark/models.py).
3. Register it in `build_model_client()` in the same file.
4. Add the required API key to [`.env.example`](.env.example).
5. Document one or two recommended model IDs in this README under a new section.
6. Update tests if parsing or provider selection logic changed.

Do not add a custom evaluation path — all providers share the loop in [`evaluator.py`](src/latamgpt_benchmark/evaluator.py).

---

## Doubleword provider notes

Integrated as an OpenAI-compatible provider using the official `openai` Python client:

- Base URL: `https://api.doubleword.ai/v1`
- API key env var: `DOUBLEWORD_API_KEY`

Recommended model IDs for this benchmark:

| Model ID | Notes |
|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | Strong mid-sized reasoning model, good price/performance |
| `openai/gpt-oss-20b` | Small open-weight baseline, low latency |
| `Qwen/Qwen3-14B-FP8` | Lighter baseline for high-volume runs |

---

## Weave logging contract

Every prediction logged to W&B Weave includes:

- Dataset row metadata
- Model output
- Latency
- Token usage
- Deterministic metrics
- Judge metrics (when `--judge` is enabled)

This is handled entirely in [`evaluator.py`](src/latamgpt_benchmark/evaluator.py).

---

## Dataset citations

**Trueque:**

```bibtex
@software{Trueque_benchmark_beta_0.1,
  title={Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture},
  author={Fuentes, Gonzalo and Arriagada, Alexandra and Henriquez, Clemente and García, M. Alexandra and LatamGPT Team},
  year={2026},
  url={https://huggingface.co/latam-gpt/Trueque-Benchmark-beta-0.1}
}
```

**CHOCLO:**

```bibtex
@dataset{choclo2026,
  title={CHOCLO: Latin American Cultural Knowledge Benchmark},
  author={Del Solar, Bianca and Carvallo, Andrés and Soto, Álvaro},
  year={2026},
  publisher={Hugging Face},
  note={Developed as part of doctoral research under the supervision of Álvaro Soto}
}
```

See [`docs/dataset-citations-and-comparison.md`](docs/dataset-citations-and-comparison.md) for a comparison of both datasets.

---

## Contributor checklist

- Code and comments in English
- Keep provider integrations simple — satisfy the `BaseModelClient` contract
- Reuse the shared evaluation loop
- Add models through `provider:model-id`
- Update `.env.example` when introducing a new credential
- Update this README when adding a provider or changing the default suite
- Run before opening a PR:

```bash
uv run --extra dev ruff check
uv run --extra dev pytest
```
