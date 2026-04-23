from pathlib import Path

from latamgpt_benchmark.config import ModelSpec
from latamgpt_benchmark.judge_only import _default_output_run


def test_default_output_run_uses_judge_slug() -> None:
    output = _default_output_run(
        Path("runs/benchmark-default"),
        ModelSpec.parse("openai:gpt-4.1-mini"),
    )
    assert output == Path("runs/benchmark-default-judge-openai__gpt-4.1-mini")
