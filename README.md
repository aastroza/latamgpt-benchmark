# latamgpt-benchmark

Batch-first evaluation pipeline for [`latam-gpt/CHOCLO`](https://huggingface.co/datasets/latam-gpt/CHOCLO) and [`latam-gpt/Trueque-Benchmark-beta-0.1`](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1) using only OpenAI and Doubleword.

## What Changed

- Only two providers remain: `openai` and `doubleword`.
- Benchmark inference is now asynchronous and batch-based.
- Judging is a separate second batch stage.
- The default OpenAI list follows the OpenAI lineage already used in CHOCLO, updated with current replacements where applicable.
- The default OpenAI list is a conservative current set: `gpt-4.1-mini`, `gpt-5.4`, `gpt-5-nano`.
- The default Doubleword list keeps one Qwen and adds Gemma plus GPT-OSS.

## Model Selection

Source rationale lives in [`docs/model-selection-2026-04-22.md`](docs/model-selection-2026-04-22.md).

Current default suite:

- `openai:gpt-4.1-mini`
- `openai:gpt-5.4`
- `openai:gpt-5-nano`
- `doubleword:Qwen/Qwen3.5-4B`
- `doubleword:google/gemma-4-31B-it`
- `doubleword:openai/gpt-oss-20b`

Default judge model:

- `openai:gpt-4.1-mini`

## Environment Variables

Required for dataset loading and batch submission:

```env
HF_TOKEN=
OPENAI_API_KEY=
DOUBLEWORD_API_KEY=
```

Optional, only if you want Weave logging during answer collection:

```env
WANDB_API_KEY=
WEAVE_PROJECT=entity/project
```

See [`.env.example`](.env.example).

## Workflow

This repo now assumes a two-stage workflow.

### 1. Submit answer batches

```bash
uv run latamgpt-benchmark submit --model-suite benchmark-default
```

This creates a run directory with:

- `config.json`
- `dataset_snapshots/`
- `batch_inputs/answers/`
- `batch_manifests/answers/`
- `answers_batches.json`

Requests are packed by model and chunked with `--max-requests-per-batch`, so batches are not submitted one row at a time.

### 2. Wait for answer batches

Refresh status once:

```bash
uv run latamgpt-benchmark status --run-dir runs/<run-name>
```

Or poll until terminal state:

```bash
uv run latamgpt-benchmark status --run-dir runs/<run-name> --wait --poll-interval-seconds 300
```

This is the waiting strategy for long jobs: submit once, then use polling with a configurable interval and optional timeout instead of blocking inside the submit step.

### 3. Collect answer batches

```bash
uv run latamgpt-benchmark collect --run-dir runs/<run-name> --wait
```

This downloads completed batch outputs and materializes the usual benchmark artifacts:

- `<dataset>__<model>.jsonl`
- `<dataset>__<model>.summary.json`
- `run_summary.json`

If `WEAVE_PROJECT` and `WANDB_API_KEY` are present, this step also logs predictions to Weave.

### 4. Submit judge batches

```bash
uv run latamgpt-judge-only submit --input-run runs/<run-name>
```

This creates a second run directory, by default:

```text
runs/<run-name>-judge-openai__gpt-4.1-mini
```

### 5. Wait for judge batches

```bash
uv run latamgpt-judge-only status --run-dir runs/<judge-run> --wait --poll-interval-seconds 300
```

### 6. Collect judge batches

```bash
uv run latamgpt-judge-only collect --run-dir runs/<judge-run> --wait
```

This writes judged copies of the answer files plus updated summaries and `run_summary.json`.

## Model Suites

Curated suites are defined in [`src/latamgpt_benchmark/model_suites.py`](src/latamgpt_benchmark/model_suites.py).

Available suites:

- `benchmark-default`
- `openai-conservative`
- `doubleword-small`
- `cost-minimal`

## Repository Layout

```text
src/latamgpt_benchmark/
|-- batching.py      # Batch polling and registry helpers
|-- cli.py           # Answer batch CLI
|-- config.py        # Defaults and run config
|-- datasets.py      # Dataset loading and normalization
|-- evaluator.py     # Answer batch submit/status/collect flow
|-- judge.py         # Judge prompt and parsing helpers
|-- judge_only.py    # Judge batch submit/status/collect flow
|-- model_suites.py  # Saved suites
|-- models.py        # Provider batch clients
|-- scoring.py       # Deterministic metrics
`-- utils.py         # JSON and filesystem helpers
```

## Provider Notes

OpenAI batches use the official Batch API and currently require `completion_window="24h"`.

Doubleword batches use the same OpenAI-compatible shape with base URL `https://api.doubleword.ai/v1`. Doubleword supports both `24h` and `1h`, but the default here is `24h` to minimize cost.

## Adding or Changing Models

Model specs still use:

```text
provider:model-id
```

To add another model from an existing provider:

1. Add it to a suite in [`src/latamgpt_benchmark/model_suites.py`](src/latamgpt_benchmark/model_suites.py), or pass it via `--model`.
2. Keep the provider as either `openai` or `doubleword`.
3. Update [`docs/model-selection-2026-04-22.md`](docs/model-selection-2026-04-22.md) and this README if the default benchmark set changes.

## Auditing Outputs

If you need to inspect a run end to end:

- Dataset loading: [`src/latamgpt_benchmark/datasets.py`](src/latamgpt_benchmark/datasets.py)
- Answer submission and materialization: [`src/latamgpt_benchmark/evaluator.py`](src/latamgpt_benchmark/evaluator.py)
- Judge prompt and parser: [`src/latamgpt_benchmark/judge.py`](src/latamgpt_benchmark/judge.py)
- Judge batch flow: [`src/latamgpt_benchmark/judge_only.py`](src/latamgpt_benchmark/judge_only.py)
- Metrics: [`src/latamgpt_benchmark/scoring.py`](src/latamgpt_benchmark/scoring.py)
- Model selection notes: [`docs/model-selection-2026-04-22.md`](docs/model-selection-2026-04-22.md)
- Dataset comparison notes: [`docs/dataset-citations-and-comparison.md`](docs/dataset-citations-and-comparison.md)

## Dataset Citations

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

## Contributor Notes

- Code and comments in English.
- Keep the pipeline simple and fail loudly when a batch response is malformed.
- Do not reintroduce per-request synchronous inference paths.
- Keep answer generation and judging as separate batch stages.
- I did not run benchmarks, tests, or linters in this change.
