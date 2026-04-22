from latamgpt_benchmark.model_suites import resolve_model_list


def test_resolve_model_list_expands_suite() -> None:
    models = resolve_model_list([], ["current-flagships"])
    assert models == [
        "openai:gpt-5.4",
        "anthropic:claude-opus-4-7",
        "gemini:gemini-2.5-pro",
        "doubleword:Qwen/Qwen3.5-397B-A17B-FP8",
    ]


def test_resolve_model_list_deduplicates_suite_and_manual_values() -> None:
    models = resolve_model_list(
        ["openai:gpt-5.4-mini", "doubleword:openai/gpt-oss-20b"],
        ["current-cost-balanced"],
    )
    assert models == [
        "openai:gpt-5.4-mini",
        "anthropic:claude-sonnet-4-6",
        "gemini:gemini-2.5-flash",
        "doubleword:Qwen/Qwen3.6-35B-A3B-FP8",
        "doubleword:openai/gpt-oss-20b",
    ]
