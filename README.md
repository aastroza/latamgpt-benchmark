# latamgpt-benchmark

Reproducible evaluation pipeline for `latam-gpt/CHOCLO` and `latam-gpt/Trueque-Benchmark-beta-0.1` using modern OpenAI, Anthropic, and Gemini models with public logging in W&B Weave.

## What it does

- Loads both datasets from Hugging Face.
- Normalizes them into a shared question/reference-answer format.
- Runs the same prompt template across multiple model providers.
- Computes deterministic metrics:
  - `normalized_exact_match`
  - `token_f1`
  - `char_similarity`
- Optionally runs an LLM judge for open-ended answers.
- Logs predictions, scores, usage, and summaries to Weave.
- Saves local artifacts in `runs/<run_name>/` for reproducibility.

## Default model set

- `openai:gpt-5.4-mini`
- `anthropic:claude-sonnet-4-6`
- `gemini:gemini-2.5-flash`

Default judge model:

- `openai:gpt-5.4-mini`

Disable the judge with `--disable-judge`.

## Requirements

- Python 3.12+
- `uv` recommended
- A W&B / Weave project configured as public if you want public visibility

## Environment variables

This repository expects a `.env` file. Fill in:

```env
WANDB_API_KEY=...
WEAVE_PROJECT=your-entity/latamgpt-benchmark
HF_TOKEN=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

### Variables you still need to add

You already have:

- `WANDB_API_KEY`
- `HF_TOKEN`

You still need:

- `WEAVE_PROJECT`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

If you do not want to use the judge model, you can omit `OPENAI_API_KEY` only if you also remove OpenAI from the evaluated model list and run with `--disable-judge`.

## Installation

```bash
uv sync --extra dev
```

## Quick start

Evaluate a small sample from both datasets:

```bash
uv run latamgpt-benchmark --max-samples 25 --shuffle --seed 7
```

Evaluate only Trueque:

```bash
uv run latamgpt-benchmark --dataset trueque --max-samples 100
```

Evaluate CHOCLO with the default models:

```bash
uv run latamgpt-benchmark --dataset choclo
```

Choose explicit model ids:

```bash
uv run latamgpt-benchmark \
  --model openai:gpt-5.4-mini \
  --model anthropic:claude-sonnet-4-6 \
  --model gemini:gemini-2.5-flash
```

Disable the LLM judge and keep only deterministic metrics:

```bash
uv run latamgpt-benchmark --disable-judge --max-samples 100
```

## Sharding large runs

CHOCLO contains more than 100k rows. To split a run across machines or processes:

```bash
uv run latamgpt-benchmark --dataset choclo --num-shards 4 --shard-index 0
uv run latamgpt-benchmark --dataset choclo --num-shards 4 --shard-index 1
uv run latamgpt-benchmark --dataset choclo --num-shards 4 --shard-index 2
uv run latamgpt-benchmark --dataset choclo --num-shards 4 --shard-index 3
```

## Local outputs

Each run creates `runs/<run_name>/` with:

- `config.json`: resolved experiment configuration
- `run_summary.json`: top-level summary for the full run
- `*.jsonl`: one row per evaluated example for each dataset/model pair
- `*.summary.json`: aggregate summary per dataset/model pair

## Public Weave logging

The code initializes Weave with `WEAVE_PROJECT` and publishes the evaluated dataset subset in that project. To make the results public:

1. Create the project in W&B if it does not already exist.
2. Set project visibility to public in the W&B UI.
3. Run the benchmark with `WEAVE_PROJECT=your-entity/latamgpt-benchmark`.

## Evaluation design

- Model input only contains the benchmark question.
- Country, category, topic, and difficulty are kept as metadata, not injected into the prompt.
- The judge compares `question`, `reference_answer`, and `model_answer`.
- The judge returns:
  - `overall_score`
  - `correctness_score`
  - `completeness_score`
  - `uncertainty_handling_score`
  - `verdict`
  - `justification`

## Reproducibility

To reproduce a run:

1. Clone the repository.
2. Fill in the `.env` file.
3. Run `uv sync --extra dev`.
4. Use the same command, seed, dataset selection, model list, and shard settings.
5. Compare both the local files in `runs/` and the public Weave run.

## Implementation notes

- OpenAI uses the `openai` Python client
- Anthropic uses the `anthropic` Python client
- Gemini uses the `google-genai` Python client
- Tracking uses `weave.EvaluationLogger`

## Run tests

```bash
uv run --extra dev pytest
```
