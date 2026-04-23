from __future__ import annotations

import json
from dataclasses import dataclass


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


def build_judge_user_prompt(question: str, reference_answer: str, model_answer: str) -> str:
    payload = {
        "question": question,
        "reference_answer": reference_answer,
        "model_answer": model_answer,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_judge_result(text: str, usage: dict[str, int | None]) -> JudgeResult:
    payload = _parse_json_payload(text)
    return JudgeResult(
        correctness_score=float(payload["correctness_score"]),
        completeness_score=float(payload["completeness_score"]),
        uncertainty_handling_score=float(payload["uncertainty_handling_score"]),
        overall_score=float(payload["overall_score"]),
        verdict=str(payload["verdict"]),
        justification=str(payload["justification"]).strip(),
        raw_response_text=text,
        usage=usage,
    )


def judge_payload(judge_result: JudgeResult) -> dict:
    return {
        "correctness_score": judge_result.correctness_score,
        "completeness_score": judge_result.completeness_score,
        "uncertainty_handling_score": judge_result.uncertainty_handling_score,
        "overall_score": judge_result.overall_score,
        "verdict": judge_result.verdict,
        "justification": judge_result.justification,
        "raw_response_text": judge_result.raw_response_text,
        "usage": judge_result.usage,
    }


def _parse_json_payload(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge response did not contain JSON: {text}")
    return json.loads(text[start : end + 1])
