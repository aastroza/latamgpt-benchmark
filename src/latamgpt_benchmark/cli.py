from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from latamgpt_benchmark.config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    BenchmarkConfig,
    ModelSpec,
)
from latamgpt_benchmark.datasets import resolve_datasets
from latamgpt_benchmark.evaluator import run_benchmark
from latamgpt_benchmark.model_suites import available_suite_names, resolve_model_list
from latamgpt_benchmark.utils import utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CHOCLO and Trueque with OpenAI, Anthropic, Gemini, Doubleword, and Weave."
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset to evaluate: choclo, trueque, or all. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model spec in provider:model-id format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-suite",
        dest="model_suites",
        action="append",
        help=(
            "Named model suite to expand into multiple --model values. "
            f"Available: {', '.join(available_suite_names())}."
        ),
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model spec in provider:model-id format. Defaults to openai:gpt-5.4-mini.",
    )
    parser.add_argument(
        "--disable-judge",
        action="store_true",
        help="Disable LLM-as-a-judge and keep only deterministic metrics.",
    )
    parser.add_argument("--weave-project", help="Weave project in entity/project format.")
    parser.add_argument("--run-name", help="Optional run name. Defaults to a UTC timestamp.")
    parser.add_argument("--output-dir", default="runs", help="Directory for local outputs.")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate.")
    parser.add_argument("--max-samples", type=int, help="Maximum number of examples per dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each dataset before sampling.")
    parser.add_argument("--seed", type=int, default=7, help="Seed used for shuffle.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index to evaluate.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens for evaluated models.",
    )
    parser.add_argument(
        "--judge-max-output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens for the judge model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for all providers.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    weave_project = args.weave_project
    if not weave_project:
        import os

        weave_project = os.getenv("WEAVE_PROJECT")
    if not weave_project:
        raise ValueError("Set --weave-project or WEAVE_PROJECT in the environment.")

    datasets = resolve_datasets(args.datasets or ["all"])
    model_values = resolve_model_list(
        model_names=args.models or DEFAULT_MODELS,
        suite_names=args.model_suites or [],
    )
    models = [ModelSpec.parse(value) for value in model_values]

    judge_model = None
    if not args.disable_judge:
        judge_value = args.judge_model or DEFAULT_JUDGE_MODEL
        judge_model = ModelSpec.parse(judge_value)

    config = BenchmarkConfig(
        datasets=datasets,
        models=models,
        judge_model=judge_model,
        weave_project=weave_project,
        output_dir=Path(args.output_dir),
        run_name=args.run_name or utc_timestamp(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        judge_prompt=DEFAULT_JUDGE_PROMPT,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_output_tokens=args.max_output_tokens,
        judge_max_output_tokens=args.judge_max_output_tokens,
        temperature=args.temperature,
    )

    run_dir = run_benchmark(config)
    print(run_dir)


if __name__ == "__main__":
    main()
