from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from latamgpt_benchmark.batching import (
    all_batches_completed,
    chunk_list,
    format_registry_summary,
    jsonl_to_dict,
    refresh_batch_registry,
)
from latamgpt_benchmark.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT, ModelSpec
from latamgpt_benchmark.judge import build_judge_user_prompt, judge_payload, parse_judge_result
from latamgpt_benchmark.models import build_batch_client, parse_batch_output_row
from latamgpt_benchmark.scoring import summarize_results
from latamgpt_benchmark.utils import (
    append_jsonl,
    ensure_directory,
    read_jsonl,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit, track, and collect batch judge runs for an existing benchmark run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit judge batches.")
    submit_parser.add_argument("--input-run", required=True, help="Completed answer run directory.")
    submit_parser.add_argument(
        "--output-run",
        help="Optional output run directory. Defaults to <input-run>-judge-<judge-model-slug>.",
    )
    submit_parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model spec in provider:model-id format.",
    )
    submit_parser.add_argument(
        "--judge-max-output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens for the judge model.",
    )
    submit_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for the judge model.",
    )
    submit_parser.add_argument(
        "--completion-window",
        default="24h",
        choices=["24h", "1h"],
        help="Batch completion window. OpenAI only supports 24h.",
    )
    submit_parser.add_argument(
        "--max-requests-per-batch",
        type=int,
        default=5000,
        help="Upper bound of requests packed into each submitted judge batch file.",
    )

    status_parser = subparsers.add_parser("status", help="Refresh and print judge batch status.")
    _add_tracking_arguments(status_parser)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Download completed judge batches and materialize judged outputs.",
    )
    _add_tracking_arguments(collect_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    if args.command == "submit":
        input_run = Path(args.input_run)
        if not input_run.exists():
            raise ValueError(f"Input run directory does not exist: {input_run}")
        judge_model = ModelSpec.parse(args.judge_model)
        output_run = Path(args.output_run) if args.output_run else _default_output_run(input_run, judge_model)
        submitted = submit_judge_batches(
            input_run=input_run,
            output_run=output_run,
            judge_model=judge_model,
            judge_prompt=DEFAULT_JUDGE_PROMPT,
            judge_max_output_tokens=args.judge_max_output_tokens,
            temperature=args.temperature,
            completion_window=args.completion_window,
            max_requests_per_batch=args.max_requests_per_batch,
        )
        print(submitted)
        return

    output_run = Path(args.run_dir)
    if args.command == "status":
        registry = refresh_judge_batches(
            output_run,
            wait=args.wait,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(format_registry_summary(registry))
        return

    collected = collect_judge_batches(
        output_run,
        wait=args.wait,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(collected)


def submit_judge_batches(
    input_run: Path,
    output_run: Path,
    judge_model: ModelSpec,
    judge_prompt: str,
    judge_max_output_tokens: int,
    temperature: float,
    completion_window: str,
    max_requests_per_batch: int,
) -> Path:
    ensure_directory(output_run)
    if (input_run / "config.json").exists():
        shutil.copy2(input_run / "config.json", output_run / "source_config.json")

    batch_inputs_dir = ensure_directory(output_run / "batch_inputs" / "judge")
    manifests_dir = ensure_directory(output_run / "batch_manifests" / "judge")

    request_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    client = build_batch_client(judge_model)
    for input_file in _input_result_files(input_run):
        for row in read_jsonl(input_file):
            custom_id = f"r{len(request_rows):07d}"
            request_rows.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": client.build_request_body(
                        system_prompt=judge_prompt,
                        user_prompt=build_judge_user_prompt(
                            question=row["question"],
                            reference_answer=row["reference_answer"],
                            model_answer=row["prediction"],
                        ),
                        max_output_tokens=judge_max_output_tokens,
                        temperature=temperature,
                    ),
                }
            )
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "source_file": input_file.name,
                    "row": row,
                }
            )

    batches: list[dict[str, Any]] = []
    for part_index, chunk in enumerate(chunk_list(request_rows, max_requests_per_batch)):
        chunk_manifest = manifest_rows[
            part_index * max_requests_per_batch : (part_index + 1) * max_requests_per_batch
        ]
        batch_name = f"judge__{judge_model.slug}__part-{part_index:03d}"
        batch_input_path = batch_inputs_dir / f"{batch_name}.jsonl"
        manifest_path = manifests_dir / f"{batch_name}.jsonl"
        write_jsonl(batch_input_path, chunk)
        write_jsonl(manifest_path, chunk_manifest)

        input_file_id = client.upload_batch_file(str(batch_input_path))
        batch_info = client.create_batch(
            input_file_id=input_file_id,
            completion_window=completion_window,
            metadata={
                "stage": "judge",
                "batch_name": batch_name,
            },
        )
        batches.append(
            {
                "batch_name": batch_name,
                "provider": judge_model.provider,
                "model": judge_model.model,
                "model_name": judge_model.name,
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
        raise ValueError(f"No judge batches were created from input run: {input_run}")

    write_json(
        output_run / "judge_only_config.json",
        {
            "input_run": str(input_run),
            "judge_model": judge_model.name,
            "judge_prompt": judge_prompt,
            "judge_max_output_tokens": judge_max_output_tokens,
            "temperature": temperature,
            "completion_window": completion_window,
            "max_requests_per_batch": max_requests_per_batch,
        },
    )
    write_json(
        output_run / "judge_batches.json",
        {
            "stage": "judge",
            "run_dir": str(output_run),
            "batches": batches,
        },
    )
    return output_run


def refresh_judge_batches(
    output_run: Path,
    wait: bool = False,
    poll_interval_seconds: int = 300,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    return refresh_batch_registry(
        output_run / "judge_batches.json",
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def collect_judge_batches(
    output_run: Path,
    wait: bool = False,
    poll_interval_seconds: int = 300,
    timeout_seconds: int | None = None,
) -> Path:
    registry = refresh_judge_batches(
        output_run,
        wait=wait,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    if not all_batches_completed(registry):
        raise ValueError(
            "Not all judge batches are completed.\n"
            f"{format_registry_summary(registry)}"
        )

    judge_config = _load_judge_config(output_run)
    batch_outputs_dir = ensure_directory(output_run / "batch_outputs" / "judge")
    judged_rows_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
            judge_result = parse_judge_result(response_row["text"], response_row["usage"])
            judged_rows_by_file[manifest_row["source_file"]].append(
                _merge_judge_row(manifest_row["row"], judge_result)
            )

    all_summaries: dict[str, dict[str, Any]] = {}
    for source_file, rows in sorted(judged_rows_by_file.items()):
        output_file = output_run / source_file
        output_file.write_text("", encoding="utf-8")
        for row in rows:
            append_jsonl(output_file, row)

        summary = summarize_results(rows)
        if rows:
            summary["dataset_name"] = rows[0]["dataset_name"]
            summary["model"] = rows[0]["model"]
        summary["judge_model"] = judge_config["judge_model"]
        summary["source_file"] = source_file
        write_json(output_run / source_file.replace(".jsonl", ".summary.json"), summary)
        all_summaries[source_file.removesuffix(".jsonl")] = summary

    write_json(output_run / "run_summary.json", all_summaries)
    return output_run


def _merge_judge_row(row: dict[str, Any], judge_result) -> dict[str, Any]:
    judged = dict(row)
    metrics = dict(row["metrics"])
    metrics.update(judge_result.metrics())
    judged["metrics"] = metrics
    judged["judge"] = judge_payload(judge_result)
    judged["judge_usage"] = judge_result.usage
    return judged


def _input_result_files(input_run: Path) -> list[Path]:
    return sorted(
        path
        for path in input_run.glob("*.jsonl")
        if "__" in path.stem and not path.name.endswith(".output.jsonl")
    )


def _load_judge_config(output_run: Path) -> dict[str, Any]:
    import json

    return json.loads((output_run / "judge_only_config.json").read_text(encoding="utf-8"))


def _add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Judge run directory created by submit.")
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


def _default_output_run(input_run: Path, judge_model: ModelSpec) -> Path:
    return input_run.parent / f"{input_run.name}-judge-{judge_model.slug}"
