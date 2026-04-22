from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s]")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    without_punctuation = NON_WORD_RE.sub(" ", without_accents)
    return WHITESPACE_RE.sub(" ", without_punctuation).strip()


def token_f1_score(reference: str, prediction: str) -> float:
    reference_tokens = normalize_text(reference).split()
    prediction_tokens = normalize_text(prediction).split()
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0

    reference_counts = _token_counts(reference_tokens)
    prediction_counts = _token_counts(prediction_tokens)
    overlap = 0
    for token, reference_count in reference_counts.items():
        overlap += min(reference_count, prediction_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def char_similarity(reference: str, prediction: str) -> float:
    return SequenceMatcher(a=normalize_text(reference), b=normalize_text(prediction)).ratio()


def deterministic_scores(reference: str, prediction: str) -> dict[str, float]:
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)
    return {
        "normalized_exact_match": float(normalized_reference == normalized_prediction),
        "token_f1": token_f1_score(reference, prediction),
        "char_similarity": char_similarity(reference, prediction),
        "answered": float(bool(normalized_prediction)),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_examples": 0,
            "metrics": {},
            "usage": {},
            "judge_usage": {},
            "breakdowns": {},
        }

    metric_values: dict[str, list[float]] = defaultdict(list)
    usage_totals: dict[str, int] = defaultdict(int)
    judge_usage_totals: dict[str, int] = defaultdict(int)
    breakdowns: dict[str, Any] = {}

    for result in results:
        for metric_name, metric_value in result["metrics"].items():
            if metric_value is None:
                continue
            metric_values[metric_name].append(float(metric_value))
        for token_name, token_value in result["usage"].items():
            if token_value is not None:
                usage_totals[token_name] += int(token_value)
        for token_name, token_value in result.get("judge_usage", {}).items():
            if token_value is not None:
                judge_usage_totals[token_name] += int(token_value)

    metadata_keys = set()
    for result in results:
        metadata_keys.update(result["metadata"].keys())

    for key in sorted(metadata_keys):
        grouped = _group_metric_summary(results, key)
        if grouped:
            breakdowns[key] = grouped

    return {
        "num_examples": len(results),
        "metrics": {name: _mean(values) for name, values in sorted(metric_values.items())},
        "usage": dict(sorted(usage_totals.items())),
        "judge_usage": dict(sorted(judge_usage_totals.items())),
        "breakdowns": breakdowns,
    }


def _group_metric_summary(results: list[dict[str, Any]], metadata_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        raw_value = result["metadata"].get(metadata_key)
        if not raw_value:
            continue
        values = [str(raw_value).strip()]
        if metadata_key == "topic":
            values = [part.strip() for part in str(raw_value).split(",") if part.strip()]
        for value in values:
            groups[value].append(result)

    summary: dict[str, Any] = {}
    for value, group_results in sorted(groups.items()):
        metric_values: dict[str, list[float]] = defaultdict(list)
        for group_result in group_results:
            for metric_name, metric_value in group_result["metrics"].items():
                if metric_value is None:
                    continue
                metric_values[metric_name].append(float(metric_value))
        summary[value] = {
            "count": len(group_results),
            "metrics": {name: _mean(values) for name, values in sorted(metric_values.items())},
        }
    return summary


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _token_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return dict(counts)
