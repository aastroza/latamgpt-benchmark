from latamgpt_benchmark.config import ModelSpec


def test_parse_doubleword_model_spec() -> None:
    spec = ModelSpec.parse("doubleword:Qwen/Qwen3.6-35B-A3B-FP8")
    assert spec.provider == "doubleword"
    assert spec.model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert spec.slug == "doubleword__Qwen_Qwen3.6-35B-A3B-FP8"
