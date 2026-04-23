from latamgpt_benchmark.model_suites import resolve_model_list


def test_resolve_model_list_expands_suite() -> None:
    models = resolve_model_list([], ["benchmark-default"])
    assert models == [
        "openai:gpt-4.1-mini",
        "openai:gpt-5.4",
        "openai:gpt-5-nano",
        "doubleword:Qwen/Qwen3.5-4B",
        "doubleword:google/gemma-4-31B-it",
        "doubleword:openai/gpt-oss-20b",
    ]


def test_resolve_model_list_deduplicates_suite_and_manual_values() -> None:
    models = resolve_model_list(
        ["openai:gpt-4.1-mini", "doubleword:openai/gpt-oss-20b"],
        ["cost-minimal"],
    )
    assert models == [
        "openai:gpt-4.1-mini",
        "doubleword:Qwen/Qwen3.5-4B",
        "doubleword:openai/gpt-oss-20b",
    ]
