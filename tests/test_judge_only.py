from pathlib import Path

from latamgpt_benchmark.config import ModelSpec
from latamgpt_benchmark.judge_only import _default_output_run


def test_default_output_run_uses_judge_slug() -> None:
    output = _default_output_run(
        Path("runs/current-recommended"),
        ModelSpec.parse("openai:gpt-5.4-mini"),
    )
    assert output == Path("runs/current-recommended-judge-openai__gpt-5.4-mini")
