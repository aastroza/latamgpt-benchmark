# Model selection review

Last reviewed: 2026-04-22

This document records the model choices now encoded in the repository after narrowing the benchmark to OpenAI and Doubleword and moving execution to batch mode.

## Selection Principle

The benchmark now follows two simple rules:

- For OpenAI, avoid the GPT models already reported in CHOCLO and include one current flagship.
- For Doubleword, keep one Qwen and add Gemma plus GPT-OSS.

## Dataset-Sourced Model Lineage

### CHOCLO

The CHOCLO dataset card explicitly reports results for:

- `GPT-4o Mini`
- `GPT-3.5 Turbo`
- `GPT-5 Mini`

It also states that benchmark construction used `GPT-3.5` during the generation pipeline.

Source:

- [CHOCLO dataset card](https://huggingface.co/datasets/latam-gpt/CHOCLO)

### Trueque

The Trueque dataset card does not provide an OpenAI baseline lineup. It only states that the current automated evaluation baseline is `Qwen2.5-Instruct (72B)`.

Source:

- [Trueque dataset card](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1)

## OpenAI

Official source for replacements:

- [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)

The replacement we use directly from that page is:

- `gpt-3.5-turbo-0125` -> `gpt-4.1-mini`

The GPT models already reported on the CHOCLO card are:

- `gpt-3.5-turbo`
- `gpt-4o-mini`
- `gpt-5-mini`

We intentionally do not carry `gpt-4o-mini` or `gpt-5-mini` in the current default suite.

For the flagship replacement, OpenAI's current model docs say: "If you're not sure where to start, use `gpt-5.4`, our flagship model for complex reasoning and coding."

Selected OpenAI models:

- `openai:gpt-4.1-mini`
- `openai:gpt-5.4`
- `openai:gpt-5-nano`

Why:

- `gpt-4.1-mini` is the explicit OpenAI replacement for the deprecated GPT-3.5 Turbo lineage on the deprecations page.
- `gpt-5.4` is the current OpenAI flagship according to the official models guide.
- `gpt-5-nano` provides the cheapest current OpenAI tier in the benchmark set.

Default judge model:

- `openai:gpt-4.1-mini`

Why:

- It is the cheapest OpenAI choice in the benchmark set with an explicit deprecation-lineage justification.
- The judge is run in its own second batch, so a smaller judge materially reduces cost.

## Doubleword

Official sources:

- [Doubleword model catalog](https://docs.doubleword.ai/inference-api/models)
- [Doubleword model pricing](https://docs.doubleword.ai/inference-api/model-pricing)

Relevant documented options:

- `Qwen/Qwen3.5-4B` is presented as a very small open 4B model for cost-sensitive workloads.
- `google/gemma-4-31B-it` is listed in the current Doubleword catalog.
- `openai/gpt-oss-20b` is listed in the current Doubleword catalog as a lower-latency open model.

Selected Doubleword models:

- `doubleword:Qwen/Qwen3.5-4B`
- `doubleword:google/gemma-4-31B-it`
- `doubleword:openai/gpt-oss-20b`

Why:

- This keeps one genuinely small Qwen in the benchmark.
- It adds Gemma and GPT-OSS as requested, so Doubleword is no longer all-Qwen.
- It still avoids the larger 35B and 397B models.

## Batch Strategy

Official sources:

- [OpenAI Batch guide](https://developers.openai.com/api/docs/guides/batch)
- [Doubleword batch inference guide](https://docs.doubleword.ai/inference-api/batch-inference)
- [Doubleword model eval example](https://github.com/doublewordai/use-cases/tree/main/model-evals)

Implementation choices in this repository:

- Answer generation is submitted first as batch requests.
- Judging is submitted second as a separate batch run over the materialized answer files.
- Requests are grouped into batch files with many rows each and split only when `max_requests_per_batch` is exceeded.
- Waiting is handled by explicit polling commands rather than by blocking inside submission.

Completion windows:

- OpenAI: `24h` only.
- Doubleword: default `24h` in this repo for cost, with optional `1h` support in the CLI.

## Saved Suites In Code

These selections are encoded in:

- [src/latamgpt_benchmark/model_suites.py](C:/Users/Alonso/Dropbox/personal/repos/latamgpt-benchmark/src/latamgpt_benchmark/model_suites.py:1)

Available suites:

- `benchmark-default`
- `openai-conservative`
- `doubleword-small`
- `cost-minimal`

## Example Commands

Submit the default answer batches:

```bash
uv run latamgpt-benchmark submit --model-suite benchmark-default
```

Track them until terminal state:

```bash
uv run latamgpt-benchmark status --run-dir runs/<run-name> --wait
```

Then submit the judge stage:

```bash
uv run latamgpt-judge-only submit --input-run runs/<run-name>
```
