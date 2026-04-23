from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from latamgpt_benchmark.config import ModelSpec
from latamgpt_benchmark.models import TERMINAL_BATCH_STATUSES, build_batch_client
from latamgpt_benchmark.utils import read_json, write_json


def chunk_list[T](values: list[T], chunk_size: int) -> list[list[T]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def refresh_batch_registry(
    registry_path: Path,
    wait: bool = False,
    poll_interval_seconds: int = 300,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        registry = read_json(registry_path)
        updated_batches: list[dict[str, Any]] = []
        for batch in registry["batches"]:
            spec = ModelSpec(provider=batch["provider"], model=batch["model"])
            info = build_batch_client(spec).retrieve_batch(batch["batch_id"])
            updated = dict(batch)
            updated["status"] = info.status
            updated["output_file_id"] = info.output_file_id
            updated["error_file_id"] = info.error_file_id
            updated["request_counts"] = info.request_counts
            updated_batches.append(updated)
        registry["batches"] = updated_batches
        write_json(registry_path, registry)

        if not wait or all_batches_terminal(registry):
            return registry
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            return registry
        time.sleep(poll_interval_seconds)


def all_batches_terminal(registry: dict[str, Any]) -> bool:
    return all(batch["status"] in TERMINAL_BATCH_STATUSES for batch in registry["batches"])


def all_batches_completed(registry: dict[str, Any]) -> bool:
    return all(batch["status"] == "completed" for batch in registry["batches"])


def format_registry_summary(registry: dict[str, Any]) -> str:
    lines = [f"stage={registry['stage']} batches={len(registry['batches'])}"]
    for batch in registry["batches"]:
        counts = batch.get("request_counts") or {}
        progress = ""
        total = counts.get("total")
        completed = counts.get("completed")
        failed = counts.get("failed")
        if total is not None or completed is not None or failed is not None:
            progress = f" total={total} completed={completed} failed={failed}"
        lines.append(
            f"{batch['batch_name']} provider={batch['provider']} model={batch['model']} "
            f"status={batch['status']}{progress}"
        )
    return "\n".join(lines)


def jsonl_to_dict(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row[key_field])] = row
    return rows
