from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import weave
from dotenv import load_dotenv
from weave import EvaluationLogger

from latamgpt_benchmark.batching import (
    all_batches_completed,
    chunk_list,
    format_registry_summary,
    jsonl_to_dict,
    refresh_batch_registry,
)
from latamgpt_benchmark.config import BenchmarkConfig, ModelSpec
from latamgpt_benchmark.datasets import (
    BenchmarkExample,
    load_benchmark_examples,
    published_dataset_name,
)
from latamgpt_benchmark.models import build_batch_client, parse_batch_output_row
from latamgpt_benchmark.scoring import deterministic_scores, summarize_results
from latamgpt_benchmark.utils import (
    append_jsonl,
    ensure_directory,
    git_commit_hash,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def submit_answer_batches(config: BenchmarkConfig) -> Path:
    load_dotenv()
    run_dir = ensure_directory(config.output_dir / config.run_name)
    write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "git_commit": git_commit_hash(Path.cwd()),
        },
    )

    snapshots_dir = ensure_directory(run_dir / "dataset_snapshots")
    batch_inputs_dir = ensure_directory(run_dir / "batch_inputs" / "answers")
    manifests_dir = ensure_directory(run_dir / "batch_manifests" / "answers")

    snapshot_examples: dict[str, list[BenchmarkExample]] = {}
    hf_token = os.getenv("HF_TOKEN")
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
        snapshot_examples[dataset_name] = examples
        write_jsonl(
            snapshots_dir / f"{dataset_name}.jsonl",
            [example.to_weave_row() for example in examples],
        )

    batches: list[dict[str, Any]] = []
    for model_spec in config.models:
        client = build_batch_client(model_spec)
        request_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for dataset_name in config.datasets:
            for example in snapshot_examples[dataset_name]:
                custom_id = f"r{len(request_rows):07d}"
                request_rows.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": client.build_request_body(
                            system_prompt=config.system_prompt,
                            user_prompt=example.question,
                            max_output_tokens=config.max_output_tokens,
                            temperature=config.temperature,
                        ),
                    }
                )
                manifest_rows.append(
                    {
                        "custom_id": custom_id,
                        "example_id": example.uid,
                        "dataset_name": example.dataset_name,
                        "split": example.split,
                        "row_id": example.row_id,
                        "question": example.question,
                        "reference_answer": example.reference_answer,
                        "metadata": example.metadata,
                        "provider": model_spec.provider,
                        "model": model_spec.model,
                        "model_name": model_spec.name,
                    }
                )

        for part_index, chunk in enumerate(chunk_list(request_rows, config.max_requests_per_batch)):
            chunk_manifest = manifest_rows[
                part_index * config.max_requests_per_batch : (part_index + 1) * config.max_requests_per_batch
            ]
            batch_name = f"answers__{model_spec.slug}__part-{part_index:03d}"
            batch_input_path = batch_inputs_dir / f"{batch_name}.jsonl"
            manifest_path = manifests_dir / f"{batch_name}.jsonl"
            write_jsonl(batch_input_path, chunk)
            write_jsonl(manifest_path, chunk_manifest)

            input_file_id = client.upload_batch_file(str(batch_input_path))
            batch_info = client.create_batch(
                input_file_id=input_file_id,
                completion_window=config.answer_completion_window,
                metadata={
                    "stage": "answers",
                    "run_name": config.run_name,
                    "batch_name": batch_name,
                },
            )
            batches.append(
                {
                    "batch_name": batch_name,
                    "provider": model_spec.provider,
                    "model": model_spec.model,
                    "model_name": model_spec.name,
                    "input_path": str(batch_input_path),
                    "manifest_path": str(manifest_path),
                    "input_file_id": input_file_id,
                    "batch_id": batch_info.batch_id,
                    "status": batch_info.status,
                    "output_file_id": batch_info.output_file_id,
                    "error_file_id": batch_info.error_file_id,
                    "request_counts": batch_info.request_counts,
                    "request_count": len(chunk),
                    }
                )

    if not batches:
        raise ValueError("No answer batches were created. Check dataset selection and sample limits.")

    registry = {
        "stage": "answers",
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "batches": batches,
    }
    write_json(run_dir / "answers_batches.json", registry)
    return run_dir


def refresh_answer_batches(
    run_dir: Path,
    wait: bool = False,
    poll_interval_seconds: int = 300,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    return refresh_batch_registry(
        run_dir / "answers_batches.json",
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def collect_answer_batches(
    run_dir: Path,
    wait: bool = False,
    poll_interval_seconds: int = 300,
    timeout_seconds: int | None = None,
) -> Path:
    registry = refresh_answer_batches(
        run_dir,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    if not all_batches_completed(registry):
        raise ValueError(
            "Not all answer batches are completed.\n"
            f"{format_registry_summary(registry)}"
        )

    config = BenchmarkConfig.from_dict(read_json(run_dir / "config.json"))
    batch_outputs_dir = ensure_directory(run_dir / "batch_outputs" / "answers")

    results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for batch in registry["batches"]:
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise ValueError(f"Batch {batch['batch_name']} did not produce an output file.")
        output_path = batch_outputs_dir / f"{batch['batch_name']}.jsonl"
        downloaded = build_batch_client(
            ModelSpec(provider=batch["provider"], model=batch["model"])
        ).download_output_file(output_file_id)
        output_path.write_text(downloaded.text, encoding="utf-8")

        manifest_rows = jsonl_to_dict(Path(batch["manifest_path"]), "custom_id")
        output_rows = jsonl_to_dict(output_path, "custom_id")
        if set(manifest_rows) != set(output_rows):
            raise ValueError(f"Output mismatch for batch {batch['batch_name']}.")

        for custom_id in manifest_rows:
            manifest_row = manifest_rows[custom_id]
            response_row = parse_batch_output_row(output_rows[custom_id])
            metrics = deterministic_scores(
                reference=manifest_row["reference_answer"],
                prediction=response_row["text"],
            )
            results_by_key[(manifest_row["dataset_name"], manifest_row["model_name"])].append(
                {
                    "example_id": manifest_row["example_id"],
                    "dataset_name": manifest_row["dataset_name"],
                    "question": manifest_row["question"],
                    "reference_answer": manifest_row["reference_answer"],
                    "metadata": manifest_row["metadata"],
                    "provider": manifest_row["provider"],
                    "model": manifest_row["model_name"],
                    "prediction": response_row["text"],
                    "latency_seconds": None,
                    "usage": response_row["usage"],
                    "metrics": metrics,
                    "judge": None,
                    "judge_usage": {},
                    "response_id": response_row["response_id"],
                    "finish_reason": response_row["finish_reason"],
                    "raw_model_name": response_row["raw_model_name"],
                }
            )

    _write_answer_outputs(run_dir, config, results_by_key)
    return run_dir


def _write_answer_outputs(
    run_dir: Path,
    config: BenchmarkConfig,
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    all_summaries: dict[str, Any] = {}
    snapshots = _load_snapshot_rows(run_dir)
    _maybe_publish_to_weave(config, snapshots, results_by_key)

    for (dataset_name, model_name), rows in sorted(results_by_key.items()):
        model_spec = ModelSpec.parse(model_name)
        result_path = run_dir / f"{dataset_name}__{model_spec.slug}.jsonl"
        summary_path = run_dir / f"{dataset_name}__{model_spec.slug}.summary.json"
        result_path.write_text("", encoding="utf-8")
        for row in rows:
            append_jsonl(result_path, row)
        summary = summarize_results(rows)
        summary["dataset_name"] = dataset_name
        summary["model"] = model_name
        summary["evaluation_name"] = f"{config.run_name}-{dataset_name}-{model_spec.slug}"
        write_json(summary_path, summary)
        all_summaries[f"{dataset_name}__{model_spec.slug}"] = summary

    write_json(run_dir / "run_summary.json", all_summaries)


def _load_snapshot_rows(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for snapshot_file in sorted((run_dir / "dataset_snapshots").glob("*.jsonl")):
        snapshots[snapshot_file.stem] = read_jsonl(snapshot_file)
    return snapshots


def _maybe_publish_to_weave(
    config: BenchmarkConfig,
    snapshots: dict[str, list[dict[str, Any]]],
    results_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    if not config.weave_project:
        return
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required to log results to Weave.")

    weave.init(config.weave_project)
    for dataset_name, rows in snapshots.items():
        _publish_dataset_subset(config=config, dataset_name=dataset_name, rows=rows)

    for (dataset_name, model_name), rows in sorted(results_by_key.items()):
        model_spec = ModelSpec.parse(model_name)
        evaluation_name = f"{config.run_name}-{dataset_name}-{model_spec.slug}"
        eval_logger = EvaluationLogger(
            name=evaluation_name,
            model=model_name,
            dataset=dataset_name,
        )
        snapshot_by_id = {row["uid"]: row for row in snapshots[dataset_name]}
        for row in rows:
            source_row = snapshot_by_id[row["example_id"]]
            pred = eval_logger.log_prediction(
                source_row,
                {
                    "answer": row["prediction"],
                    "provider": model_spec.provider,
                    "model": model_spec.model,
                    "latency_seconds": row["latency_seconds"],
                    "usage": row["usage"],
                    "response_id": row["response_id"],
                    "finish_reason": row["finish_reason"],
                    "raw_model_name": row["raw_model_name"],
                    "judge": row["judge"],
                },
            )
            for metric_name, metric_value in row["metrics"].items():
                pred.log_score(metric_name, metric_value)
            pred.finish()
        summary = summarize_results(rows)
        summary["dataset_name"] = dataset_name
        summary["model"] = model_name
        summary["evaluation_name"] = evaluation_name
        eval_logger.log_summary(summary)


def _publish_dataset_subset(
    config: BenchmarkConfig,
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    dataset_name_for_weave = published_dataset_name(
        dataset_name=dataset_name,
        split=config.split,
        sample_count=len(rows),
        shuffle=config.shuffle,
        seed=config.seed,
        num_shards=config.num_shards,
        shard_index=config.shard_index,
    )
    weave.publish(weave.Dataset(name=dataset_name_for_weave, rows=rows))
