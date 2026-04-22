from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


DATASET_ALIASES = {
    "choclo": "choclo",
    "trueque": "trueque",
    "all": "all",
}


@dataclass(frozen=True)
class BenchmarkExample:
    dataset_name: str
    split: str
    row_id: int
    question: str
    reference_answer: str
    metadata: dict[str, Any]

    @property
    def uid(self) -> str:
        return f"{self.dataset_name}:{self.split}:{self.row_id}"

    def to_weave_row(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "row_id": self.row_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }


def resolve_datasets(selection: list[str]) -> list[str]:
    if not selection:
        return ["choclo", "trueque"]
    expanded: list[str] = []
    for value in selection:
        normalized = DATASET_ALIASES.get(value.lower(), value.lower())
        if normalized == "all":
            expanded.extend(["choclo", "trueque"])
            continue
        if normalized not in {"choclo", "trueque"}:
            raise ValueError(f"Unsupported dataset '{value}'. Use choclo, trueque, or all.")
        expanded.append(normalized)
    seen: set[str] = set()
    deduped: list[str] = []
    for name in expanded:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def dataset_hf_id(dataset_name: str) -> str:
    if dataset_name == "choclo":
        return "latam-gpt/CHOCLO"
    if dataset_name == "trueque":
        return "latam-gpt/Trueque-Benchmark-beta-0.1"
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def load_benchmark_examples(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 7,
    num_shards: int = 1,
    shard_index: int = 0,
    hf_token: str | None = None,
) -> list[BenchmarkExample]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be between 0 and num_shards - 1.")

    hf_id = dataset_hf_id(dataset_name)
    dataset = load_dataset(hf_id, split=split, token=hf_token)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if num_shards > 1:
        dataset = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    if max_samples is not None:
        sample_count = min(max_samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    rows = list(dataset)
    return [_normalize_row(dataset_name=dataset_name, split=split, row_id=i, row=row) for i, row in enumerate(rows)]


def published_dataset_name(
    dataset_name: str,
    split: str,
    sample_count: int,
    shuffle: bool,
    seed: int,
    num_shards: int,
    shard_index: int,
) -> str:
    parts = [
        "latamgpt",
        dataset_name,
        split,
        f"n{sample_count}",
        f"shuffle{int(shuffle)}",
        f"seed{seed}",
        f"shard{shard_index}-of-{num_shards}",
    ]
    return "-".join(parts)


def _normalize_row(
    dataset_name: str,
    split: str,
    row_id: int,
    row: dict[str, Any],
) -> BenchmarkExample:
    if dataset_name == "choclo":
        question = _clean_text(row["Question"])
        answer = _clean_text(row["Answer"])
        metadata = {
            "entity": _clean_text(row["Entity"]),
            "country": _clean_text(row["Country"]),
            "category": _clean_text(row["Category"]),
            "difficulty": _clean_text(row["Difficulty"]),
        }
    elif dataset_name == "trueque":
        question = _clean_text(row["question"])
        answer = _clean_text(row["reference_answer"])
        metadata = {
            "country": _clean_text(row["country"]),
            "topic": _clean_text(row["topic"]),
        }
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    return BenchmarkExample(
        dataset_name=dataset_name,
        split=split,
        row_id=row_id,
        question=question,
        reference_answer=answer,
        metadata=metadata,
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
