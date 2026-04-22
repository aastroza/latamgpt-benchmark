from latamgpt_benchmark.datasets import _normalize_row


def test_normalize_choclo_row() -> None:
    example = _normalize_row(
        dataset_name="choclo",
        split="train",
        row_id=3,
        row={
            "Entity": "Valparaiso",
            "Country": "Chile",
            "Category": "geography",
            "Difficulty": "FACIL",
            "Question": "Que es Valparaiso?",
            "Answer": "Una ciudad portuaria de Chile.",
        },
    )
    assert example.uid == "choclo:train:3"
    assert example.question == "Que es Valparaiso?"
    assert example.reference_answer == "Una ciudad portuaria de Chile."
    assert example.metadata["country"] == "Chile"


def test_normalize_trueque_row() -> None:
    example = _normalize_row(
        dataset_name="trueque",
        split="train",
        row_id=5,
        row={
            "question": "Que es la corpachada?",
            "reference_answer": "Una ceremonia andina.",
            "country": "Argentina",
            "topic": "Pueblos-Originarios, Celebraciones",
        },
    )
    assert example.uid == "trueque:train:5"
    assert example.reference_answer == "Una ceremonia andina."
    assert example.metadata["topic"] == "Pueblos-Originarios, Celebraciones"
