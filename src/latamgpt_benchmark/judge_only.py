from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from latamgpt_benchmark.config import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT, ModelSpec
from latamgpt_benchmark.judge import JudgeResult, run_judge
from latamgpt_benchmark.models import build_model_client
from latamgpt_benchmark.scoring import summarize_results
from latamgpt_benchmark.utils import ensure_directory, read_jsonl, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run judge-only scoring on an existing benchmark run directory."
    )
    parser.add_argument(
        "--input-run",
        required=True,
        help="Existing run directory containing *.jsonl outputs.",
    )
    parser.add_argument(
        "--output-run",
        help="Optional output run directory. Defaults to <input-run>-judge-<judge-model-slug>.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Judge model spec in provider:model-id format.",
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
        help="Generation temperature for the judge model.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv()

    input_run = Path(args.input_run)
    if not input_run.exists():
        raise ValueError(f"Input run directory does not exist: {input_run}")

    judge_model = ModelSpec.parse(args.judge_model)
    output_run = Path(args.output_run) if args.output_run else _default_output_run(input_run, judge_model)
    run_judge_only(
        input_run=input_run,
        output_run=output_run,
        judge_model=judge_model,
        judge_prompt=DEFAULT_JUDGE_PROMPT,
        judge_max_output_tokens=args.judge_max_output_tokens,
        temperature=args.temperature,
    )
    print(output_run)


def run_judge_only(
    input_run: Path,
    output_run: Path,
    judge_model: ModelSpec,
    judge_prompt: str,
    judge_max_output_tokens: int,
    temperature: float,
) -> Path:
    ensure_directory(output_run)
    judge_client = build_model_client(
        judge_model,
        max_output_tokens=judge_max_output_tokens,
        temperature=temperature,
    )

    config_path = input_run / "config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_run / "source_config.json")

    all_summaries: dict[str, dict] = {}

    for input_file in sorted(input_run.glob("*.jsonl")):
        rows = read_jsonl(input_file)
        judged_rows = [_judge_row(row, judge_client, judge_prompt) for row in rows]

        output_file = output_run / input_file.name
        output_file.write_text("", encoding="utf-8")
        for row in judged_rows:
            _append_row(output_file, row)

        summary = summarize_results(judged_rows)
        if judged_rows:
            summary["dataset_name"] = judged_rows[0]["dataset_name"]
            summary["model"] = judged_rows[0]["model"]
        summary["judge_model"] = judge_model.name
        summary["source_file"] = input_file.name
        summary_file = output_run / input_file.name.replace(".jsonl", ".summary.json")
        write_json(summary_file, summary)
        all_summaries[input_file.stem] = summary

    write_json(
        output_run / "judge_only_config.json",
        {
            "input_run": str(input_run),
            "judge_model": judge_model.name,
            "judge_max_output_tokens": judge_max_output_tokens,
            "temperature": temperature,
        },
    )
    write_json(output_run / "run_summary.json", all_summaries)
    return output_run


def _judge_row(row: dict, judge_client, judge_prompt: str) -> dict:
    judged = dict(row)
    judge_result = run_judge(
        judge_client=judge_client,
        judge_prompt=judge_prompt,
        question=row["question"],
        reference_answer=row["reference_answer"],
        model_answer=row["prediction"],
    )
    metrics = dict(row["metrics"])
    metrics.update(judge_result.metrics())
    judged["metrics"] = metrics
    judged["judge"] = _judge_payload(judge_result)
    judged["judge_usage"] = judge_result.usage
    return judged


def _judge_payload(judge_result: JudgeResult) -> dict:
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


def _append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _default_output_run(input_run: Path, judge_model: ModelSpec) -> Path:
    return input_run.parent / f"{input_run.name}-judge-{judge_model.slug}"
