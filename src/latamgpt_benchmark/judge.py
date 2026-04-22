from __future__ import annotations

import json
from dataclasses import dataclass

from latamgpt_benchmark.models import BaseModelClient


@dataclass(frozen=True)
class JudgeResult:
    correctness_score: float
    completeness_score: float
    uncertainty_handling_score: float
    overall_score: float
    verdict: str
    justification: str
    raw_response_text: str
    usage: dict[str, int | None]

    def metrics(self) -> dict[str, float]:
        return {
            "judge_correctness": self.correctness_score,
            "judge_completeness": self.completeness_score,
            "judge_uncertainty_handling": self.uncertainty_handling_score,
            "judge_overall": self.overall_score,
        }


def run_judge(
    judge_client: BaseModelClient,
    judge_prompt: str,
    question: str,
    reference_answer: str,
    model_answer: str,
) -> JudgeResult:
    prompt = _build_judge_prompt(
        question=question,
        reference_answer=reference_answer,
        model_answer=model_answer,
    )
    response = judge_client.generate(system_prompt=judge_prompt, user_prompt=prompt)
    payload = _parse_json_payload(response.text)
    return JudgeResult(
        correctness_score=float(payload["correctness_score"]),
        completeness_score=float(payload["completeness_score"]),
        uncertainty_handling_score=float(payload["uncertainty_handling_score"]),
        overall_score=float(payload["overall_score"]),
        verdict=str(payload["verdict"]),
        justification=str(payload["justification"]).strip(),
        raw_response_text=response.text,
        usage=response.usage,
    )


def _build_judge_prompt(question: str, reference_answer: str, model_answer: str) -> str:
    payload = {
        "question": question,
        "reference_answer": reference_answer,
        "model_answer": model_answer,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json_payload(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge response did not contain JSON: {text}")
    return json.loads(text[start : end + 1])
