from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .adapters import rubricarrow_items
from .schema import RubricCriterion

JUDGE_REVISION = "a76e107c09ee6d599c51bddc1d799e834bdc5064"
TOP_LOGPROBS = 10


def build_conversation_text(instruction: str, response: str) -> str:
    return f"user: {instruction}\n\nassistant: {response}"


def create_pointwise_judge_prompt(
    conversation: str, rubric_items: str, criterion_count: int,
) -> str:
    if criterion_count < 1:
        raise ValueError("criterion_count must be positive")
    schema_lines: list[str] = []
    for index in range(1, criterion_count + 1):
        suffix = "," if index < criterion_count else ""
        schema_lines.extend((
            f'  "explanation_{index}": "...",',
            f'  "criteria_met_{index}": true{suffix}',
        ))
    exact_schema = "{\n" + "\n".join(schema_lines) + "\n}"
    return f"""
Your job is to look at a conversation and a set of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_items}

# Instructions
Return a json object with exactly {2 * criterion_count} top-level fields. For each rubric item i (starting from 1), keys must be exactly "explanation_i" and "criteria_met_i":
- The "explanation_i" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met_i" field should be a boolean indicating (true/false) whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Final Output Format (a single JSON object, not an array)
{exact_schema}

# Final instruction
Return just the json object with exactly the fields shown above. Do not add fields for rubric items that are not shown. Do not include any other text in the response.
""".strip()


def judge_prompt(instruction: str, response: str, criteria: Sequence[RubricCriterion]) -> str:
    return create_pointwise_judge_prompt(
        build_conversation_text(instruction, response), rubricarrow_items(criteria), len(criteria)
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value = json.loads(cleaned.replace("True", "true").replace("False", "false"))
        except json.JSONDecodeError as error:
            raise ValueError("RubricARROW output is not a JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("RubricARROW output must be one JSON object")
    return value


@dataclass(frozen=True)
class CriterionProbability:
    index: int
    p_true: float
    p_false: float
    z: float
    criteria_met: bool
    explanation: str


def strict_criterion_probabilities(
    text: str,
    probabilities: dict[int, dict[str, float]],
    criterion_count: int,
) -> tuple[CriterionProbability, ...]:
    parsed = parse_json_object(text)
    output: list[CriterionProbability] = []
    for index in range(1, criterion_count + 1):
        met = parsed.get(f"criteria_met_{index}")
        explanation = parsed.get(f"explanation_{index}")
        if not isinstance(met, bool) or not isinstance(explanation, str):
            raise ValueError(f"RubricARROW parser failure at criterion {index}: missing typed fields")
        pair = probabilities.get(index)
        if pair is None or "true_prob" not in pair or "false_prob" not in pair:
            raise ValueError(f"RubricARROW parser failure at criterion {index}: missing true/false log-prob")
        p_true = float(pair["true_prob"])
        p_false = float(pair["false_prob"])
        if not math.isfinite(p_true) or not math.isfinite(p_false) or p_true < 0 or p_false < 0:
            raise ValueError(f"RubricARROW parser failure at criterion {index}: invalid probability")
        if p_true == 0.0 or p_false == 0.0:
            raise ValueError(f"RubricARROW parser failure at criterion {index}: zero-probability branch")
        z = min(1.0, max(0.0, (1.0 + p_true - p_false) / 2.0))
        output.append(CriterionProbability(index, p_true, p_false, z, met, explanation))
    return tuple(output)


def probabilities_from_openai_logprobs(entries: Sequence[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Extract true/false alternatives at each generated criteria_met value token."""
    prefix = ""
    result: dict[int, dict[str, float]] = {}
    for entry in entries:
        token = str(entry.get("token", ""))
        pending = re.search(r'"criteria_met_(\d+)"\s*:\s*$', prefix)
        if pending:
            values: dict[str, float] = {}
            candidates = entry.get("top_logprobs") or []
            for candidate in candidates:
                candidate_token = str(candidate.get("token", ""))
                normalized = candidate_token.strip().lower()
                if normalized in ("true", "false"):
                    logprob = float(candidate["logprob"])
                    if math.isfinite(logprob):
                        values[f"{normalized}_prob"] = math.exp(logprob)
            if values:
                result[int(pending.group(1))] = values
        prefix = (prefix + token)[-800:]
    return result


def criterion_value_token_positions(tokens: Sequence[str]) -> dict[int, int]:
    """Locate the first generated token after every criteria_met_i JSON key."""
    prefix = ""
    result: dict[int, int] = {}
    for position, token in enumerate(tokens):
        pending = re.search(r'"criteria_met_(\d+)"\s*:\s*$', prefix)
        if pending:
            index = int(pending.group(1))
            if index in result:
                raise ValueError(f"duplicate criteria_met_{index} value position")
            result[index] = position
        prefix = (prefix + str(token))[-800:]
    return result
