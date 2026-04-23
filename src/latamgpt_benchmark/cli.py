from __future__ import annotations

import argparse
import os
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
from latamgpt_benchmark.evaluator import (
    collect_answer_batches,
    refresh_answer_batches,
    submit_answer_batches,
)
from latamgpt_benchmark.model_suites import available_suite_names, resolve_model_list
from latamgpt_benchmark.utils import utc_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit, track, and collect LATAMGPT benchmark batches with OpenAI and Doubleword."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit answer-generation batches.")
    _add_submit_arguments(submit_parser)

    status_parser = subparsers.add_parser("status", help="Refresh and print answer batch status.")
    _add_tracking_arguments(status_parser)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Download completed answer batches and materialize benchmark outputs.",
    )
    _add_tracking_arguments(collect_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    if args.command == "submit":
        run_dir = submit_answer_batches(_benchmark_config_from_args(args))
        print(run_dir)
        return

    run_dir = Path(args.run_dir)
    if args.command == "status":
        registry = refresh_answer_batches(
            run_dir,
            wait=args.wait,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        from latamgpt_benchmark.batching import format_registry_summary

        print(format_registry_summary(registry))
        return

    collected = collect_answer_batches(
        run_dir,
        wait=args.wait,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(collected)


def _add_submit_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--weave-project", help="Optional Weave project in entity/project format.")
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
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for all providers.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        choices=["24h", "1h"],
        help="Batch completion window. OpenAI only supports 24h.",
    )
    parser.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=5000,
        help="Upper bound of requests packed into each submitted batch file.",
    )


def _add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Run directory created by submit.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the batches reach a terminal state before returning.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=300,
        help="Polling interval used with --wait.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Optional timeout used with --wait.",
    )


def _benchmark_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    weave_project = args.weave_project or os.getenv("WEAVE_PROJECT")
    datasets = resolve_datasets(args.datasets or ["all"])
    model_values = resolve_model_list(
        model_names=args.models or DEFAULT_MODELS,
        suite_names=args.model_suites or [],
    )
    models = [ModelSpec.parse(value) for value in model_values]

    judge_model = ModelSpec.parse(DEFAULT_JUDGE_MODEL)
    return BenchmarkConfig(
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
        judge_max_output_tokens=256,
        temperature=args.temperature,
        answer_completion_window=args.completion_window,
        judge_completion_window="24h",
        max_requests_per_batch=args.max_requests_per_batch,
    )
