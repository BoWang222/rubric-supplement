from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np
import torch

from .schema import (
    LOCAL_EXACT_SCORER_PROTOCOL,
    RubricCriterion,
    SCORER_CRITERION_CHUNK_SIZE,
    SCORER_SEED,
)
from .scorer import (
    criterion_value_token_positions,
    judge_prompt,
    strict_criterion_probabilities,
)


def criterion_chunk_spans(
    criterion_count: int, max_size: int = SCORER_CRITERION_CHUNK_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Split criteria into balanced, ordered chunks no larger than max_size.

    Balancing avoids pathological one- or two-item tail prompts while preserving
    the registered maximum chunk size and exact criterion order.
    """
    if criterion_count < 1 or max_size < 1:
        raise ValueError("criterion_count and max_size must be positive")
    chunk_count = math.ceil(criterion_count / max_size)
    base, extra = divmod(criterion_count, chunk_count)
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(chunk_count):
        size = base + int(index < extra)
        spans.append((start, start + size))
        start += size
    if start != criterion_count or any(end - begin > max_size for begin, end in spans):
        raise AssertionError("balanced criterion chunking violated its contract")
    return tuple(spans)


def _instruction(prompt_messages: Sequence[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role']}: {message['content']}" for message in prompt_messages
    )


def _candidate_ids(tokenizer, observed_token: str, value: str) -> list[int]:
    whitespace = observed_token[: len(observed_token) - len(observed_token.lstrip())]
    candidates = tokenizer.encode(f"{whitespace}{value}", add_special_tokens=False)
    if not candidates:
        raise ValueError(f"tokenizer produced no IDs for Boolean candidate {value!r}")
    return list(map(int, candidates))


def _forced_sequence_logprob(model, prefix_ids: torch.Tensor, candidate_ids: Sequence[int]) -> float:
    candidate = torch.tensor([list(candidate_ids)], dtype=torch.long, device=prefix_ids.device)
    full = torch.cat((prefix_ids, candidate), dim=1)
    with torch.no_grad():
        logits = model(input_ids=full, use_cache=False).logits[0]
    start = prefix_ids.shape[1] - 1
    selected = torch.log_softmax(
        logits[start:start + len(candidate_ids)].float(), dim=-1
    ).gather(-1, candidate[0].unsqueeze(-1)).squeeze(-1)
    return float(selected.double().sum().item())


def exact_boolean_branch_probabilities(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    generated: torch.Tensor,
    criterion_count: int,
) -> dict[int, dict[str, float]]:
    """Score both JSON Boolean branches even when one lies outside top_logprobs."""
    token_texts = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in generated.tolist()
    ]
    positions = criterion_value_token_positions(token_texts)
    expected = set(range(1, criterion_count + 1))
    if set(positions) != expected:
        raise ValueError(
            f"RubricARROW Boolean positions mismatch: expected={sorted(expected)}, "
            f"found={sorted(positions)}"
        )
    full_sequence = torch.cat((input_ids, generated.unsqueeze(0)), dim=1)
    with torch.no_grad():
        # This forward pass is deliberate: generate().scores has already passed
        # through temperature/top-p processors and can contain artificial -inf.
        raw_logits = model(input_ids=full_sequence, use_cache=False).logits[0]
    result: dict[int, dict[str, float]] = {}
    for index in sorted(positions):
        position = positions[index]
        observed = token_texts[position]
        prefix = torch.cat((input_ids, generated[:position].unsqueeze(0)), dim=1)
        probabilities: dict[str, float] = {}
        for value in ("true", "false"):
            candidate_ids = _candidate_ids(tokenizer, observed, value)
            if len(candidate_ids) == 1:
                logprob = float(torch.log_softmax(
                    raw_logits[input_ids.shape[1] + position - 1].float(), dim=-1
                )[candidate_ids[0]].item())
            else:
                logprob = _forced_sequence_logprob(model, prefix, candidate_ids)
            # Python float can retain probabilities far below FP32's exp range.
            probabilities[f"{value}_prob"] = math.exp(max(logprob, -744.0))
            probabilities[f"{value}_logprob"] = logprob
        result[index] = probabilities
    return result


def score_one_response(
    model,
    tokenizer,
    prompt_messages: Sequence[dict[str, str]],
    response: str,
    criteria: Sequence[RubricCriterion],
    *,
    max_new_tokens: int = 512,
    scorer_seed: int = SCORER_SEED,
) -> tuple[list[float], dict[str, Any]]:
    rendered = judge_prompt(_instruction(prompt_messages), response, criteria)
    if scorer_seed != SCORER_SEED:
        raise ValueError(f"scorer_seed is frozen to {SCORER_SEED}")
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    ).to(next(model.parameters()).device)
    device = input_ids.device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(scorer_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(scorer_seed)
        output = model.generate(
            input_ids,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    generated = output.sequences[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    probabilities = exact_boolean_branch_probabilities(
        model, tokenizer, input_ids, generated, len(criteria)
    )
    try:
        parsed = strict_criterion_probabilities(text, probabilities, len(criteria))
    except ValueError as error:
        raise ValueError(
            f"{error}; extracted_probabilities={probabilities}; generated_text={text[:1200]!r}"
        ) from error
    return [item.z for item in parsed], {
        "text": text,
        "scorer_protocol": LOCAL_EXACT_SCORER_PROTOCOL,
        "scorer_seed": scorer_seed,
        "probabilities": {
            str(item.index): {
                "p_true": item.p_true, "p_false": item.p_false, "z": item.z,
                "true_logprob": probabilities[item.index]["true_logprob"],
                "false_logprob": probabilities[item.index]["false_logprob"],
            }
            for item in parsed
        },
    }


def _score_response_in_chunks(
    model,
    tokenizer,
    prompt_messages: Sequence[dict[str, str]],
    response: str,
    criteria: Sequence[RubricCriterion],
) -> tuple[list[float], dict[str, Any]]:
    """Score long rubrics without allowing JSON generation to truncate later items."""
    scores: list[float] = []
    chunks: list[dict[str, Any]] = []
    for start, end in criterion_chunk_spans(len(criteria)):
        chunk_scores, chunk_record = score_one_response(
            model, tokenizer, prompt_messages, response, criteria[start:end]
        )
        scores.extend(chunk_scores)
        chunks.append({
            "criterion_start": start + 1,
            "criterion_end": end,
            "record": chunk_record,
        })
    if len(scores) != len(criteria):
        raise AssertionError("chunked RubricARROW scoring lost criteria")
    return scores, {
        "scorer_protocol": LOCAL_EXACT_SCORER_PROTOCOL,
        "criterion_count": len(criteria),
        "criterion_chunk_size": SCORER_CRITERION_CHUNK_SIZE,
        "chunks": chunks,
    }


def rubric_distances_for_pairs(
    model,
    tokenizer,
    prompt_messages: Sequence[dict[str, str]],
    criteria: Sequence[RubricCriterion],
    paired_responses: Sequence[dict[str, Any]],
) -> tuple[tuple[float, ...], list[dict[str, Any]]]:
    by_direction: dict[int, list[float]] = {}
    records: list[dict[str, Any]] = []
    for pair in paired_responses:
        base, base_record = _score_response_in_chunks(
            model, tokenizer, prompt_messages, str(pair["base_response"]), criteria
        )
        perturbed, perturbed_record = _score_response_in_chunks(
            model, tokenizer, prompt_messages, str(pair["perturbed_response"]), criteria
        )
        distance = float(np.linalg.norm(np.asarray(base) - np.asarray(perturbed)))
        direction = int(pair["direction"])
        by_direction.setdefault(direction, []).append(distance)
        records.append({
            **pair, "base_scores": base, "perturbed_scores": perturbed,
            "rubric_distance": distance, "base_judge": base_record,
            "perturbed_judge": perturbed_record,
        })
    expected = list(range(max(by_direction, default=-1) + 1))
    if sorted(by_direction) != expected:
        raise ValueError("judge pairs do not cover consecutive Fisher directions")
    distances = tuple(float(np.mean(by_direction[index])) for index in expected)
    return distances, records


def build_batched_judge_requests(
    context_id: str,
    prompt_messages: Sequence[dict[str, str]],
    criteria: Sequence[RubricCriterion],
    paired_responses: Sequence[dict[str, Any]],
    *,
    max_chunk_size: int = SCORER_CRITERION_CHUNK_SIZE,
    request_namespace: str = "primary",
) -> list[dict[str, Any]]:
    """Flatten every response/criterion chunk into scheduler-independent requests."""
    requests: list[dict[str, Any]] = []
    instruction = _instruction(prompt_messages)
    for pair_index, pair in enumerate(paired_responses):
        for side in ("base", "perturbed"):
            response = str(pair[f"{side}_response"])
            for start, end in criterion_chunk_spans(len(criteria), max_chunk_size):
                request_id = (
                    f"{context_id}:{pair_index}:{side}:{start}:{request_namespace}"
                )
                requests.append({
                    "request_id": request_id,
                    "context_id": context_id,
                    "pair_index": pair_index,
                    "side": side,
                    "criterion_start": start,
                    "criterion_end": end,
                    "criterion_count": end - start,
                    "rendered_user_content": judge_prompt(
                        instruction, response, criteria[start:end]
                    ),
                })
    return requests


def _generation_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["request_id"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise ValueError("vLLM generations contain duplicate request_id")
    return result


def score_generated_requests_exact(
    model,
    tokenizer,
    requests: Sequence[dict[str, Any]],
    generations: Sequence[dict[str, Any]],
    *,
    batch_size: int = 4,
) -> dict[str, dict[str, Any]]:
    """Batch teacher-force vLLM outputs and score both raw Boolean branches.

    Generation and exact scoring are deliberately separate.  vLLM provides
    parallel decoding; this function uses raw, unwarped Hugging Face logits for
    the mathematical criterion probabilities.
    """
    if batch_size < 1:
        raise ValueError("exact score batch_size must be positive")
    generation_by_id = _generation_index(generations)
    expected_ids = {str(request["request_id"]) for request in requests}
    if set(generation_by_id) != expected_ids:
        raise ValueError("vLLM generation request coverage mismatch")
    device = next(model.parameters()).device
    padding_id = tokenizer.pad_token_id
    if padding_id is None:
        padding_id = tokenizer.eos_token_id
    results: dict[str, dict[str, Any]] = {}

    for offset in range(0, len(requests), batch_size):
        micro_requests = list(requests[offset:offset + batch_size])
        prepared: list[dict[str, Any]] = []
        for request in micro_requests:
            request_id = str(request["request_id"])
            generation = generation_by_id[request_id]
            try:
                prompt_ids = list(map(int, generation["prompt_token_ids"]))
                generated_ids = list(map(int, generation["generated_token_ids"]))
                if not prompt_ids or not generated_ids:
                    raise ValueError("empty prompt or generation token sequence")
                token_texts = [
                    tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    for token_id in generated_ids
                ]
                positions = criterion_value_token_positions(token_texts)
                criterion_count = int(request["criterion_count"])
                expected = set(range(1, criterion_count + 1))
                if set(positions) != expected:
                    raise ValueError(
                        f"RubricARROW Boolean positions mismatch: expected={sorted(expected)}, "
                        f"found={sorted(positions)}"
                    )
                prepared.append({
                    "request": request,
                    "generation": generation,
                    "prompt_ids": prompt_ids,
                    "generated_ids": generated_ids,
                    "token_texts": token_texts,
                    "positions": positions,
                    "full_ids": prompt_ids + generated_ids,
                })
            except Exception as error:
                results[request_id] = {
                    "error": f"{type(error).__name__}: {error}",
                    "text": str(generation.get("text", "")),
                }
        if not prepared:
            continue
        max_length = max(len(item["full_ids"]) for item in prepared)
        input_ids = torch.full(
            (len(prepared), max_length), int(padding_id),
            dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, item in enumerate(prepared):
            length = len(item["full_ids"])
            input_ids[row_index, :length] = torch.tensor(
                item["full_ids"], dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1
        with torch.no_grad():
            raw_logits = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False,
            ).logits
        for row_index, item in enumerate(prepared):
            request = item["request"]
            generation = item["generation"]
            request_id = str(request["request_id"])
            try:
                probabilities: dict[int, dict[str, float]] = {}
                prompt_length = len(item["prompt_ids"])
                generated_tensor = torch.tensor(
                    item["generated_ids"], dtype=torch.long, device=device
                )
                prompt_tensor = torch.tensor(
                    [item["prompt_ids"]], dtype=torch.long, device=device
                )
                for index in sorted(item["positions"]):
                    position = item["positions"][index]
                    observed = item["token_texts"][position]
                    prefix = torch.cat(
                        (prompt_tensor, generated_tensor[:position].unsqueeze(0)), dim=1
                    )
                    branch: dict[str, float] = {}
                    for value in ("true", "false"):
                        candidate_ids = _candidate_ids(tokenizer, observed, value)
                        if len(candidate_ids) == 1:
                            logprob = float(torch.log_softmax(
                                raw_logits[
                                    row_index, prompt_length + position - 1
                                ].float(), dim=-1,
                            )[candidate_ids[0]].item())
                        else:
                            logprob = _forced_sequence_logprob(
                                model, prefix, candidate_ids
                            )
                        branch[f"{value}_prob"] = math.exp(max(logprob, -744.0))
                        branch[f"{value}_logprob"] = logprob
                    probabilities[index] = branch
                text = str(generation["text"]).strip()
                parsed = strict_criterion_probabilities(
                    text, probabilities, int(request["criterion_count"])
                )
                results[request_id] = {
                    "scores": [item.z for item in parsed],
                    "record": {
                        "text": text,
                        "finish_reason": generation.get("finish_reason"),
                        "scorer_protocol": LOCAL_EXACT_SCORER_PROTOCOL,
                        "scorer_seed": SCORER_SEED,
                        "probabilities": {
                            str(parsed_item.index): {
                                "p_true": parsed_item.p_true,
                                "p_false": parsed_item.p_false,
                                "z": parsed_item.z,
                                "true_logprob": probabilities[parsed_item.index]["true_logprob"],
                                "false_logprob": probabilities[parsed_item.index]["false_logprob"],
                            }
                            for parsed_item in parsed
                        },
                    },
                }
            except Exception as error:
                results[request_id] = {
                    "error": f"{type(error).__name__}: {error}",
                    "text": str(generation.get("text", ""))[:1200],
                }
        del raw_logits, input_ids, attention_mask
    return results


def rubric_distances_from_batched_scores(
    criteria: Sequence[RubricCriterion],
    paired_responses: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    scored: dict[str, dict[str, Any]],
) -> tuple[tuple[float, ...], list[dict[str, Any]]]:
    """Reassemble criterion order and compute the documented vector L2."""
    request_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for request in requests:
        request_groups.setdefault(
            (int(request["pair_index"]), str(request["side"])), []
        ).append(request)
    by_direction: dict[int, list[float]] = {}
    records: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(paired_responses):
        sides: dict[str, list[float]] = {}
        judges: dict[str, dict[str, Any]] = {}
        for side in ("base", "perturbed"):
            chunks = sorted(
                request_groups.get((pair_index, side), []),
                key=lambda item: int(item["criterion_start"]),
            )
            side_scores: list[float] = []
            chunk_records: list[dict[str, Any]] = []
            for request in chunks:
                outcome = scored[str(request["request_id"])]
                if "error" in outcome:
                    raise ValueError(
                        f"{request['request_id']}: {outcome['error']}; "
                        f"generated_text={outcome.get('text', '')!r}"
                    )
                side_scores.extend(map(float, outcome["scores"]))
                chunk_records.append({
                    "criterion_start": int(request["criterion_start"]) + 1,
                    "criterion_end": int(request["criterion_end"]),
                    "record": outcome["record"],
                })
            if len(side_scores) != len(criteria):
                raise AssertionError("batched RubricARROW scoring lost criterion order")
            sides[side] = side_scores
            judges[side] = {
                "scorer_protocol": LOCAL_EXACT_SCORER_PROTOCOL,
                "criterion_count": len(criteria),
                "criterion_chunk_size": SCORER_CRITERION_CHUNK_SIZE,
                "chunks": chunk_records,
            }
        distance = float(np.linalg.norm(
            np.asarray(sides["base"]) - np.asarray(sides["perturbed"])
        ))
        direction = int(pair["direction"])
        by_direction.setdefault(direction, []).append(distance)
        records.append({
            **pair,
            "base_scores": sides["base"],
            "perturbed_scores": sides["perturbed"],
            "rubric_distance": distance,
            "base_judge": judges["base"],
            "perturbed_judge": judges["perturbed"],
        })
    expected = list(range(max(by_direction, default=-1) + 1))
    if sorted(by_direction) != expected:
        raise ValueError("judge pairs do not cover consecutive Fisher directions")
    return tuple(float(np.mean(by_direction[index])) for index in expected), records
