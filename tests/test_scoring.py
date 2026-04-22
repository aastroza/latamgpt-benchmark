from latamgpt_benchmark.scoring import deterministic_scores, normalize_text, token_f1_score


def test_normalize_text_removes_accents_and_punctuation() -> None:
    assert normalize_text("¡Canción, Árbol!") == "cancion arbol"


def test_token_f1_score_handles_partial_overlap() -> None:
    score = token_f1_score(
        "una ciudad portuaria de chile",
        "ciudad de chile",
    )
    assert 0.6 < score < 0.8


def test_deterministic_scores_for_exact_match() -> None:
    scores = deterministic_scores(
        reference="Una ceremonia andina.",
        prediction="una ceremonia andina",
    )
    assert scores["normalized_exact_match"] == 1.0
    assert scores["answered"] == 1.0
