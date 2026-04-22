from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import weave
from dotenv import load_dotenv
from weave import EvaluationLogger

from latamgpt_benchmark.config import BenchmarkConfig, ModelSpec
from latamgpt_benchmark.datasets import (
    BenchmarkExample,
    load_benchmark_examples,
    published_dataset_name,
)
from latamgpt_benchmark.judge import JudgeResult, run_judge
from latamgpt_benchmark.models import BaseModelClient, build_model_client
from latamgpt_benchmark.scoring import deterministic_scores, summarize_results
from latamgpt_benchmark.utils import append_jsonl, ensure_directory, git_commit_hash, read_jsonl, write_json


def run_benchmark(config: BenchmarkConfig) -> Path:
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required to log results to Weave.")

    run_dir = ensure_directory(config.output_dir / config.run_name)
    write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "git_commit": git_commit_hash(Path.cwd()),
        },
    )

    weave.init(config.weave_project)

    all_summaries: dict[str, Any] = {}
    hf_token = os.getenv("HF_TOKEN")
    judge_client = (
        build_model_client(
            config.judge_model,
            max_output_tokens=config.judge_max_output_tokens,
            temperature=config.temperature,
        )
        if config.judge_model
        else None
    )

    for dataset_name in config.datasets:
        examples = load_benchmark_examples(
            dataset_name=dataset_name,
            split=config.split,
            max_samples=config.max_samples,
            shuffle=config.shuffle,
            seed=config.seed,
            num_shards=config.num_shards,
            shard_index=config.shard_index,
            hf_token=hf_token,
        )
        _publish_dataset_subset(config=config, dataset_name=dataset_name, examples=examples)

        for model_spec in config.models:
            model_client = build_model_client(
                model_spec,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
            )
            pair_summary = _run_single_evaluation(
                config=config,
                dataset_name=dataset_name,
                examples=examples,
                model_spec=model_spec,
                model_client=model_client,
                judge_client=judge_client,
                run_dir=run_dir,
            )
            key = f"{dataset_name}__{model_spec.slug}"
            all_summaries[key] = pair_summary

    write_json(run_dir / "run_summary.json", all_summaries)
    return run_dir


def _run_single_evaluation(
    config: BenchmarkConfig,
    dataset_name: str,
    examples: list[BenchmarkExample],
    model_spec: ModelSpec,
    model_client: BaseModelClient,
    judge_client: BaseModelClient | None,
    run_dir: Path,
) -> dict[str, Any]:
    result_path = run_dir / f"{dataset_name}__{model_spec.slug}.jsonl"
    summary_path = run_dir / f"{dataset_name}__{model_spec.slug}.summary.json"
    evaluation_name = f"{config.run_name}-{dataset_name}-{model_spec.slug}"
    eval_logger = EvaluationLogger(
        name=evaluation_name,
        model=model_spec.name,
        dataset=dataset_name,
    )

    for example in examples:
        model_response = model_client.generate(
            system_prompt=config.system_prompt,
            user_prompt=example.question,
        )
        metrics = deterministic_scores(
            reference=example.reference_answer,
            prediction=model_response.text,
        )
        judge_result = None
        if judge_client is not None:
            judge_result = run_judge(
                judge_client=judge_client,
                judge_prompt=config.judge_prompt,
                question=example.question,
                reference_answer=example.reference_answer,
                model_answer=model_response.text,
            )
            metrics.update(judge_result.metrics())

        pred = eval_logger.log_prediction(
            example.to_weave_row(),
            {
                "answer": model_response.text,
                "provider": model_spec.provider,
                "model": model_spec.model,
                "latency_seconds": model_response.latency_seconds,
                "usage": model_response.usage,
                "response_id": model_response.response_id,
                "finish_reason": model_response.finish_reason,
                "raw_model_name": model_response.raw_model_name,
                "judge": _judge_payload(judge_result),
            },
        )
        for metric_name, metric_value in metrics.items():
            pred.log_score(metric_name, metric_value)
        pred.finish()

        append_jsonl(
            result_path,
            {
                "example_id": example.uid,
                "dataset_name": example.dataset_name,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "metadata": example.metadata,
                "model": model_spec.name,
                "prediction": model_response.text,
                "latency_seconds": model_response.latency_seconds,
                "usage": model_response.usage,
                "metrics": metrics,
                "judge": _judge_payload(judge_result),
                "judge_usage": judge_result.usage if judge_result else {},
            },
        )

    results = read_jsonl(result_path)
    summary = summarize_results(results)
    summary["dataset_name"] = dataset_name
    summary["model"] = model_spec.name
    summary["evaluation_name"] = evaluation_name
    eval_logger.log_summary(summary)
    write_json(summary_path, summary)
    return summary


def _publish_dataset_subset(
    config: BenchmarkConfig,
    dataset_name: str,
    examples: list[BenchmarkExample],
) -> None:
    if not examples:
        return
    dataset_name_for_weave = published_dataset_name(
        dataset_name=dataset_name,
        split=config.split,
        sample_count=len(examples),
        shuffle=config.shuffle,
        seed=config.seed,
        num_shards=config.num_shards,
        shard_index=config.shard_index,
    )
    weave.publish(
        weave.Dataset(
            name=dataset_name_for_weave,
            rows=[example.to_weave_row() for example in examples],
        )
    )


def _judge_payload(judge_result: JudgeResult | None) -> dict[str, Any] | None:
    if judge_result is None:
        return None
    return {
        "correctness_score": judge_result.correctness_score,
        "completeness_score": judge_result.completeness_score,
        "uncertainty_handling_score": judge_result.uncertainty_handling_score,
        "overall_score": judge_result.overall_score,
        "verdict": judge_result.verdict,
        "justification": judge_result.justification,
        "raw_response_text": judge_result.raw_response_text,
        "usage": judge_result.usage,
    }
