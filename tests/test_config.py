from latamgpt_benchmark.config import ModelSpec


def test_parse_doubleword_model_spec() -> None:
    spec = ModelSpec.parse("doubleword:Qwen/Qwen3.5-4B")
    assert spec.provider == "doubleword"
    assert spec.model == "Qwen/Qwen3.5-4B"
    assert spec.slug == "doubleword__Qwen_Qwen3.5-4B"


def test_parse_rejects_removed_providers() -> None:
    try:
        ModelSpec.parse("anthropic:claude-sonnet-4-6")
    except ValueError as error:
        assert "Use openai or doubleword" in str(error)
    else:
        raise AssertionError("Expected removed provider to fail parsing.")
