from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .math import (
    adaptive_gram_svd,
    direction_coefficients,
    exact_parameter_perturbation,
    full_softmax_jsd,
)
from .schema import FisherCalibrationConfig


@dataclass(frozen=True)
class ProbeResult:
    responses: tuple[str, ...]
    singular_values: tuple[float, ...]
    response_distances: tuple[float, ...]
    epsilon_values: tuple[float, ...]
    response_distances_by_epsilon: tuple[tuple[float, ...], ...]
    paired_responses: tuple[dict[str, Any], ...]
    numerical_rank: int
    directions_used: int
    peak_reserved_gib: float
    sampled_response_tokens: int
    paired_generation_tokens: int


def _model_body(model):
    body = getattr(model, "model", None)
    if body is None:
        raise ValueError("probe model does not expose .model")
    if not hasattr(body, "layers") and hasattr(body, "model"):
        body = body.model
    if not hasattr(body, "layers") or not hasattr(body, "norm"):
        raise ValueError("probe model must expose transformer layers and final RMSNorm")
    return body


def selected_probe_parameters(model, final_blocks: int) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for group in selected_probe_parameter_groups(model, final_blocks)
        for parameter in group
    )


def selected_probe_parameter_groups(
    model, final_blocks: int
) -> tuple[tuple[torch.nn.Parameter, ...], ...]:
    body = _model_body(model)
    layers = list(body.layers)
    if len(layers) < final_blocks:
        raise ValueError(f"probe has only {len(layers)} layers, cannot select {final_blocks}")
    groups: list[tuple[torch.nn.Parameter, ...]] = []
    seen: set[int] = set()
    for module in [*layers[-final_blocks:], body.norm]:
        parameters: list[torch.nn.Parameter] = []
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
        if parameters:
            groups.append(tuple(parameters))
    if not groups:
        raise ValueError("selected Fisher probe parameter set is empty")
    return tuple(groups)


def _freeze_to_probe_subspace(
    model, final_blocks: int
) -> tuple[tuple[torch.nn.Parameter, ...], ...]:
    """Freeze every parameter outside the registered native probe subspace."""
    groups = selected_probe_parameter_groups(model, final_blocks)
    selected = {id(parameter) for group in groups for parameter in group}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    return groups


def _prompt_ids(tokenizer, prompt_messages: Sequence[dict[str, str]], device: torch.device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        list(prompt_messages), tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return torch.tensor([ids], dtype=torch.long, device=device)


def _one_generation(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    devices = [prompt_ids.device.index] if prompt_ids.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        if prompt_ids.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        output = model.generate(
            prompt_ids,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    completion = output[0, prompt_ids.shape[1]:].tolist()
    if not completion or completion[-1] != tokenizer.eos_token_id:
        completion.append(int(tokenizer.eos_token_id))
    text_ids = completion[:-1] if completion[-1] == tokenizer.eos_token_id else completion
    return completion, tokenizer.decode(text_ids, skip_special_tokens=True).strip()


def _score_bank(
    model,
    prompt: torch.Tensor,
    completions: Sequence[Sequence[int]],
    *,
    response_chunk: int = 2,
) -> torch.Tensor:
    device = prompt.device
    scores: list[torch.Tensor] = []
    prompt_length = prompt.shape[1]
    for start in range(0, len(completions), response_chunk):
        completion_chunk = completions[start:start + response_chunk]
        sequences = [
            torch.cat((prompt[0], torch.tensor(value, device=device, dtype=torch.long)))
            for value in completion_chunk
        ]
        lengths = [sequence.numel() for sequence in sequences]
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        attention = torch.arange(padded.shape[1], device=device).unsqueeze(0) < torch.tensor(
            lengths, device=device
        ).unsqueeze(1)
        output = model(input_ids=padded, attention_mask=attention, use_cache=False)
        for local_index, completion in enumerate(completion_chunk):
            targets = torch.tensor(completion, device=device, dtype=torch.long)
            positions = torch.arange(
                prompt_length - 1, prompt_length - 1 + len(completion), device=device
            )
            # Softmax is independent across sequence positions. Select the
            # completion rows before promoting logits to FP32 so long prompts
            # do not materialize a [batch, full_sequence, vocab] FP32 tensor.
            # This is the same token log-likelihood and gradient, with a much
            # smaller temporary allocation on capacity-constrained GPUs.
            completion_logits = output.logits[local_index, positions]
            token_logps = F.log_softmax(completion_logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            scores.append(token_logps.mean())
    return torch.stack(scores)


def _raw_gram(
    scores: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    response_chunk: int = 4,
) -> torch.Tensor:
    m = scores.numel()
    # Accumulate C H C tensor-by-tensor. Each parameter's response-gradient
    # bank lives only on CPU in FP32; GPU VJPs are response-chunked so a large
    # projection matrix never creates an M-times-parameter CUDA allocation.
    centered = torch.zeros((m, m), dtype=torch.float64, device="cpu")
    eye = torch.eye(m, dtype=scores.dtype, device=scores.device)
    for parameter in parameters:
        chunks: list[tuple[int, torch.Tensor]] = []
        for start in range(0, m, response_chunk):
            stop = min(m, start + response_chunk)
            try:
                gradient = torch.autograd.grad(
                    scores, parameter, grad_outputs=eye[start:stop], retain_graph=True,
                    create_graph=False, is_grads_batched=True, allow_unused=False,
                )[0]
            except RuntimeError:
                gradient = torch.stack([
                    torch.autograd.grad(
                        scores[index], parameter, retain_graph=True,
                        create_graph=False, allow_unused=False,
                    )[0]
                    for index in range(start, stop)
                ])
            chunks.append((start, gradient.detach().to("cpu", dtype=torch.float32).flatten(start_dim=1)))
            del gradient
            torch.cuda.empty_cache()
        mean = sum(chunk.sum(0) for _, chunk in chunks) / m
        for left_position, (left_start, left_raw) in enumerate(chunks):
            left = left_raw - mean
            left_stop = left_start + left.shape[0]
            for right_start, right_raw in chunks[left_position:]:
                right = right_raw - mean
                right_stop = right_start + right.shape[0]
                block = (left @ right.T).double()
                centered[left_start:left_stop, right_start:right_stop].add_(block)
                if right_start != left_start:
                    centered[right_start:right_stop, left_start:left_stop].add_(block.T)
                del right, block
            del left
        del chunks, mean
    return centered


def _centered_parameter_banks(
    banks: Sequence[Sequence[tuple[int, torch.Tensor]]],
    m: int,
) -> torch.Tensor:
    centered = torch.zeros((m, m), dtype=torch.float64, device="cpu")
    for chunks in banks:
        mean = sum(chunk.sum(0) for _, chunk in chunks) / m
        for left_position, (left_start, left_raw) in enumerate(chunks):
            left = left_raw - mean
            left_stop = left_start + left.shape[0]
            for right_start, right_raw in chunks[left_position:]:
                right = right_raw - mean
                right_stop = right_start + right.shape[0]
                block = (left @ right.T).double()
                centered[left_start:left_stop, right_start:right_stop].add_(block)
                if right_start != left_start:
                    centered[right_start:right_stop, left_start:left_stop].add_(block.T)
                del right, block
            del left
    return centered


def _centered_gram_from_gradient_banks(
    banks: Sequence[torch.Tensor],
    *,
    feature_chunk: int = 1_048_576,
) -> torch.Tensor:
    """Accumulate the exact sample-space centered Gram from CPU FP32 banks."""
    if not banks or feature_chunk < 1:
        raise ValueError("gradient banks must be non-empty and feature_chunk positive")
    m = banks[0].shape[0]
    if any(bank.device.type != "cpu" or bank.dtype != torch.float32 for bank in banks):
        raise ValueError("gradient banks must live on CPU in FP32")
    if any(bank.ndim != 2 or bank.shape[0] != m for bank in banks):
        raise ValueError("gradient banks must share one two-dimensional response axis")
    gram = torch.zeros((m, m), dtype=torch.float64, device="cpu")
    for bank in banks:
        for start in range(0, bank.shape[1], feature_chunk):
            raw = bank[:, start:start + feature_chunk]
            centered = raw - raw.mean(dim=0, keepdim=True)
            gram.add_((centered @ centered.T).double())
            del centered
    return gram


def _collect_gradient_banks(
    model,
    prompt: torch.Tensor,
    completions: Sequence[Sequence[int]],
    parameters: Sequence[torch.nn.Parameter],
    *,
    response_chunk: int = 2,
) -> tuple[torch.Tensor, ...]:
    """Compute every real response gradient once and retain it on CPU in FP32."""
    m = len(completions)
    if m < 2 or not parameters or response_chunk < 1:
        raise ValueError("gradient collection requires responses and probe parameters")
    banks = tuple(
        torch.empty((m, parameter.numel()), dtype=torch.float32, device="cpu")
        for parameter in parameters
    )
    for start in range(0, m, response_chunk):
        stop = min(m, start + response_chunk)
        scores = _score_bank(
            model, prompt, completions[start:stop], response_chunk=response_chunk
        )
        for local_index in range(stop - start):
            gradients = torch.autograd.grad(
                scores[local_index], tuple(parameters),
                retain_graph=local_index + 1 < stop - start,
                create_graph=False, allow_unused=False,
            )
            for bank, gradient in zip(banks, gradients):
                bank[start + local_index].copy_(
                    gradient.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                )
            del gradients
        del scores
        gc.collect()
        torch.cuda.empty_cache()
    return banks


def _direction_from_gradient_banks(
    banks: Sequence[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    coefficients: torch.Tensor,
    *,
    response_chunk: int = 2,
) -> tuple[torch.Tensor, ...]:
    """Reconstruct one direction with the registered BF16 chunk accumulation."""
    if len(banks) != len(parameters) or response_chunk < 1:
        raise ValueError("gradient bank and parameter counts disagree")
    coefficients_cpu = coefficients.detach().to(device="cpu", dtype=torch.float32)
    if any(bank.shape[0] != coefficients_cpu.numel() for bank in banks):
        raise ValueError("direction coefficients disagree with the response bank")
    directions: list[torch.Tensor] = []
    for bank, parameter in zip(banks, parameters):
        target = torch.zeros(parameter.shape, dtype=parameter.dtype, device="cpu")
        for start in range(0, coefficients_cpu.numel(), response_chunk):
            stop = min(coefficients_cpu.numel(), start + response_chunk)
            partial = torch.mv(
                bank[start:stop].T, coefficients_cpu[start:stop]
            ).reshape(parameter.shape)
            target.add_(partial.to(dtype=parameter.dtype))
            del partial
        directions.append(target)
    return tuple(directions)


def _streaming_group_gram(
    model,
    prompt: torch.Tensor,
    completions: Sequence[Sequence[int]],
    parameter_groups: Sequence[Sequence[torch.nn.Parameter]],
    *,
    response_chunk: int = 2,
) -> torch.Tensor:
    """Recompute small response graphs per module group and accumulate only MxM state."""
    m = len(completions)
    raw = torch.zeros((m, m), dtype=torch.float64, device="cpu")
    for group in parameter_groups:
        banks: list[list[tuple[int, torch.Tensor]]] = [[] for _ in group]
        for start in range(0, m, response_chunk):
            stop = min(m, start + response_chunk)
            scores = _score_bank(
                model, prompt, completions[start:stop], response_chunk=response_chunk
            )
            # FlashAttention has no is_grads_batched backward rule. Use scalar
            # VJPs deliberately; probing the unsupported path first frees graph
            # state and makes a fallback unsafe.
            per_response = [
                torch.autograd.grad(
                    scores[index], tuple(group),
                    retain_graph=index + 1 < stop - start,
                    create_graph=False, allow_unused=False,
                )
                for index in range(stop - start)
            ]
            gradients = tuple(
                torch.stack([values[param_index] for values in per_response])
                for param_index in range(len(group))
            )
            for param_index, gradient in enumerate(gradients):
                banks[param_index].append((
                    start,
                    gradient.detach().to("cpu", dtype=torch.float32).flatten(start_dim=1),
                ))
            del scores, gradients
            gc.collect()
            torch.cuda.empty_cache()
        raw.add_(_centered_parameter_banks(banks, m))
        del banks
        gc.collect()
    return raw


def _streaming_direction(
    model,
    prompt: torch.Tensor,
    completions: Sequence[Sequence[int]],
    parameters: Sequence[torch.nn.Parameter],
    coefficients: torch.Tensor,
    *,
    response_chunk: int = 2,
) -> tuple[torch.Tensor, ...]:
    direction = [torch.zeros_like(parameter) for parameter in parameters]
    for start in range(0, len(completions), response_chunk):
        stop = min(len(completions), start + response_chunk)
        scores = _score_bank(model, prompt, completions[start:stop], response_chunk=response_chunk)
        gradients = torch.autograd.grad(
            scores, tuple(parameters),
            grad_outputs=coefficients[start:stop].to(scores.device, scores.dtype),
            retain_graph=False, create_graph=False, allow_unused=False,
        )
        with torch.no_grad():
            for target, gradient in zip(direction, gradients):
                target.add_(gradient)
        del scores, gradients
        gc.collect()
        torch.cuda.empty_cache()
    return tuple(direction)


def _completion_logits(model, prompt: torch.Tensor, completion: Sequence[int]) -> torch.Tensor:
    full = torch.cat((prompt[0], torch.tensor(completion, device=prompt.device, dtype=torch.long))).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=full, use_cache=False).logits[0]
    start = prompt.shape[1] - 1
    return logits[start:start + len(completion)]


def calibrate_probe_context(
    model,
    tokenizer,
    prompt_messages: Sequence[dict[str, str]],
    config: FisherCalibrationConfig,
    *,
    epsilon: float,
    seed: int,
    epsilon_schedule: Sequence[float] | None = None,
) -> ProbeResult:
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Fisher calibration probe requires CUDA")
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    prompt = _prompt_ids(tokenizer, prompt_messages, device)
    # Freezing parameters outside the registered probe subspace changes neither
    # the forward pass nor any selected gradient; it only removes irrelevant
    # autograd bookkeeping.
    parameter_groups = _freeze_to_probe_subspace(model, config.final_blocks)
    parameters = tuple(parameter for group in parameter_groups for parameter in group)
    bank = [
        _one_generation(
            model, tokenizer, prompt, seed=seed + index,
            temperature=config.decoding_temperature, top_p=config.decoding_top_p,
            max_new_tokens=config.max_new_tokens,
        )
        for index in range(config.responses_m)
    ]
    completion_ids = [item[0] for item in bank]
    response_texts = [item[1] for item in bank]
    gradient_banks = _collect_gradient_banks(
        model, prompt, completion_ids, parameters
    )
    raw = _centered_gram_from_gradient_banks(gradient_banks)
    singular, response_vectors, numerical_rank = adaptive_gram_svd(
        raw, max_k=config.directions_k, min_k=config.min_directions_k
    )
    directions_used = int(singular.numel())
    coefficients = direction_coefficients(singular.float(), response_vectors.float())
    del raw, response_vectors
    gc.collect()
    torch.cuda.empty_cache()
    epsilon_values = tuple(float(value) for value in (epsilon_schedule or (epsilon,)))
    if epsilon not in epsilon_values or any(value < 0 for value in epsilon_values):
        raise ValueError("epsilon schedule must be non-negative and contain nominal epsilon")
    nominal_epsilon_index = epsilon_values.index(epsilon)
    response_distances_by_epsilon: list[list[float]] = [
        [] for _ in epsilon_values
    ]
    # The nominal model and calibration response bank are identical for every
    # Fisher direction. Keep one FP16 cache on GPU so full-softmax JSD never
    # transfers full-vocabulary logits to CPU.
    base_logits = [
        _completion_logits(model, prompt, completion).to(dtype=torch.float16)
        for completion in completion_ids
    ]
    paired: list[dict[str, Any]] = []
    paired_generation_tokens = 0
    paired_seeds = [
        seed + 100_000 + repeat for repeat in range(config.paired_generations_r)
    ]
    # The base arm is independent of Fisher direction.  Generate it once and
    # reuse the same random draw for every plus-only perturbed direction.
    base_pairs = [
        _one_generation(
            model, tokenizer, prompt, seed=paired_seed,
            temperature=config.decoding_temperature, top_p=config.decoding_top_p,
            max_new_tokens=config.max_new_tokens,
        )
        for paired_seed in paired_seeds
    ]
    paired_generation_tokens += sum(len(base_pair[0]) for base_pair in base_pairs)
    for direction_index in range(directions_used):
        direction = _direction_from_gradient_banks(
            gradient_banks, parameters, coefficients[:, direction_index]
        )
        gc.collect()
        torch.cuda.empty_cache()
        perturbed_pairs = None
        for epsilon_index, epsilon_value in enumerate(epsilon_values):
            if epsilon_value == 0.0:
                # Identical parameters imply identical logits exactly; the gate
                # still applies the registered FP32 tolerance as its floor.
                response_distances_by_epsilon[epsilon_index].append(0.0)
                continue
            with exact_parameter_perturbation(parameters, direction, epsilon_value):
                if epsilon_index == nominal_epsilon_index:
                    perturbed_pairs = [
                        _one_generation(
                            model, tokenizer, prompt,
                            seed=paired_seed,
                            temperature=config.decoding_temperature, top_p=config.decoding_top_p,
                            max_new_tokens=config.max_new_tokens,
                        )
                        for paired_seed in paired_seeds
                    ]
                jsd_values: list[float] = []
                for base, completion in zip(base_logits, completion_ids):
                    perturbed = _completion_logits(
                        model, prompt, completion
                    ).to(dtype=torch.float16)
                    jsd_values.append(full_softmax_jsd(base, perturbed))
                    del perturbed
            response_distances_by_epsilon[epsilon_index].append(
                float(sum(jsd_values) / len(jsd_values))
            )
            gc.collect()
            torch.cuda.empty_cache()
        if perturbed_pairs is None:
            raise RuntimeError("nominal epsilon perturbation did not run")
        for repeat, (base_pair, perturbed_pair) in enumerate(zip(base_pairs, perturbed_pairs)):
            paired_generation_tokens += len(perturbed_pair[0])
            paired.append({
                "direction": direction_index,
                "repeat": repeat,
                "seed": paired_seeds[repeat],
                "base_response": base_pair[1],
                "perturbed_response": perturbed_pair[1],
            })
        del direction
        gc.collect()
        torch.cuda.empty_cache()
    del base_logits, gradient_banks
    return ProbeResult(
        responses=tuple(response_texts),
        singular_values=tuple(float(value) for value in singular.cpu().tolist()),
        response_distances=tuple(response_distances_by_epsilon[nominal_epsilon_index]),
        epsilon_values=epsilon_values,
        response_distances_by_epsilon=tuple(
            tuple(values) for values in response_distances_by_epsilon
        ),
        paired_responses=tuple(paired),
        numerical_rank=numerical_rank,
        directions_used=directions_used,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        sampled_response_tokens=sum(len(completion) for completion in completion_ids),
        paired_generation_tokens=paired_generation_tokens,
    )
