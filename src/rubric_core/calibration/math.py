from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau


def centered_gram(raw_gram: torch.Tensor) -> torch.Tensor:
    if raw_gram.ndim != 2 or raw_gram.shape[0] != raw_gram.shape[1]:
        raise ValueError("raw Gram must be square")
    if not torch.isfinite(raw_gram).all():
        raise ValueError("raw Gram contains non-finite values")
    size = raw_gram.shape[0]
    identity = torch.eye(size, dtype=raw_gram.dtype, device=raw_gram.device)
    center = identity - torch.full_like(raw_gram, 1.0 / size)
    result = center @ ((raw_gram + raw_gram.T) / 2.0) @ center
    return (result + result.T) / 2.0


def gram_svd(
    raw_gram: torch.Tensor,
    k: int,
    *,
    relative_rank_threshold: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    gram = centered_gram(raw_gram.to(torch.float64))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    singular = torch.sqrt(eigenvalues)
    if singular.numel() == 0 or singular[0] <= 0:
        raise ValueError("rank_deficient: centered Gram has no positive mode")
    numerical_rank = int((singular / singular[0] > relative_rank_threshold).sum().item())
    if numerical_rank < k:
        ratios = (singular[: min(12, singular.numel())] / singular[0]).tolist()
        raise ValueError(
            f"rank_deficient: need {k} modes, found {numerical_rank}; leading ratios={ratios}"
        )
    singular = singular[:k]
    vectors = eigenvectors[:, :k]
    # Fix the SVD sign by making the largest-magnitude response coordinate positive.
    for index in range(k):
        pivot = int(torch.argmax(torch.abs(vectors[:, index])).item())
        if vectors[pivot, index] < 0:
            vectors[:, index].neg_()
    return singular, vectors


def adaptive_gram_svd(
    raw_gram: torch.Tensor,
    *,
    max_k: int,
    min_k: int,
    relative_rank_threshold: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return every stable leading mode up to max_k, with a minimum-rank gate."""
    if not 2 <= min_k <= max_k:
        raise ValueError("adaptive SVD requires 2 <= min_k <= max_k")
    gram = centered_gram(raw_gram.to(torch.float64))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    full_singular = torch.sqrt(eigenvalues)
    if full_singular.numel() == 0 or full_singular[0] <= 0:
        raise ValueError("rank_below_min: centered Gram has no positive mode")
    ratios = full_singular / full_singular[0]
    # Row centering removes exactly one degree of freedom.  Cap the reported
    # rank at M-1 so a tiny positive eigensolver residual cannot reintroduce
    # the removed all-ones response direction.
    numerical_rank = min(
        int((ratios > relative_rank_threshold).sum().item()), raw_gram.shape[0] - 1
    )
    if numerical_rank < min_k:
        leading = ratios[: min(12, ratios.numel())].tolist()
        raise ValueError(
            f"rank_below_min: need at least {min_k} modes, found {numerical_rank}; "
            f"leading ratios={leading}"
        )
    directions_used = min(max_k, numerical_rank)
    singular = full_singular[:directions_used]
    vectors = eigenvectors[:, :directions_used]
    for index in range(directions_used):
        pivot = int(torch.argmax(torch.abs(vectors[:, index])).item())
        if vectors[pivot, index] < 0:
            vectors[:, index].neg_()
    return singular, vectors, numerical_rank


def direction_coefficients(singular_values: torch.Tensor, response_vectors: torch.Tensor) -> torch.Tensor:
    if response_vectors.shape[1] != singular_values.numel():
        raise ValueError("singular values and response vectors disagree")
    return response_vectors / singular_values.unsqueeze(0)


def accumulate_raw_gram(raw_gram: torch.Tensor, batched_parameter_grads: torch.Tensor) -> None:
    if batched_parameter_grads.shape[0] != raw_gram.shape[0]:
        raise ValueError("gradient response axis does not match Gram")
    flattened = batched_parameter_grads.detach().to(torch.float32).flatten(start_dim=1)
    raw_gram.add_(flattened @ flattened.T)


def full_softmax_jsd(left_logits: torch.Tensor, right_logits: torch.Tensor, chunk_tokens: int = 32) -> float:
    if left_logits.shape != right_logits.shape or left_logits.ndim < 2:
        raise ValueError("JSD logits must have the same token-by-vocabulary shape")
    left = left_logits.reshape(-1, left_logits.shape[-1])
    right = right_logits.reshape(-1, right_logits.shape[-1])
    total = torch.zeros((), dtype=torch.float64, device=left.device)
    count = 0
    for start in range(0, left.shape[0], chunk_tokens):
        l_log = torch.log_softmax(left[start:start + chunk_tokens].float(), dim=-1)
        r_log = torch.log_softmax(right[start:start + chunk_tokens].float(), dim=-1)
        log_m = torch.logaddexp(l_log, r_log) - math.log(2.0)
        l_prob = l_log.exp()
        r_prob = r_log.exp()
        js = 0.5 * ((l_prob * (l_log - log_m)).sum(-1) + (r_prob * (r_log - log_m)).sum(-1))
        total += js.double().sum()
        count += js.numel()
    value = float((total / max(count, 1)).item())
    return min(math.log(2.0), max(0.0, value))


def epsilon_correctness_gate(
    epsilon_values: Sequence[float],
    distances_by_epsilon: Sequence[Sequence[Sequence[float]]],
) -> dict[str, float | bool | int | list[int]]:
    """Validate zero, half, nominal, and double-epsilon response-JSD behavior."""
    values = np.asarray(epsilon_values, dtype=np.float64)
    if values.shape != (4,) or not np.allclose(values, [0.0, values[2] / 2.0, values[2], values[2] * 2.0]):
        raise ValueError("correctness epsilon schedule must be [0, epsilon/2, epsilon, 2*epsilon]")
    contexts = [np.asarray(item, dtype=np.float64) for item in distances_by_epsilon]
    if not contexts:
        raise ValueError("correctness distances require at least one context")
    direction_counts = [int(item.shape[1]) if item.ndim == 2 else 0 for item in contexts]
    if any(item.ndim != 2 or item.shape[0] != 4 or item.shape[1] < 2 for item in contexts):
        raise ValueError(
            "correctness distances must contain one [4, k_i] array per context with k_i >= 2"
        )
    if any(not np.isfinite(item).all() or (item < 0).any() for item in contexts):
        raise ValueError("correctness distances must be finite and non-negative")
    zero, half, nominal, double = (
        np.concatenate([item[index] for item in contexts]) for index in range(4)
    )
    fp32_tolerance = float(np.finfo(np.float32).eps)
    zero_p99 = float(np.quantile(zero, 0.99))
    response_tie_floor = max(zero_p99, fp32_tolerance)
    monotonic = (half <= nominal + response_tie_floor) & (
        nominal <= double + response_tie_floor
    )
    monotonic_rate = float(monotonic.mean())
    nominal_median = float(np.median(nominal))
    double_p95 = float(np.quantile(double, 0.95))
    separation_ratio = nominal_median / response_tie_floor
    passed = monotonic_rate >= 0.90 and separation_ratio >= 100.0 and double_p95 < 0.1
    return {
        "passed": bool(passed),
        "local_monotonic_rate": monotonic_rate,
        "zero_jsd_p99": zero_p99,
        "response_tie_floor": response_tie_floor,
        "nominal_jsd_median": nominal_median,
        "nominal_to_zero_floor_ratio": separation_ratio,
        "double_epsilon_jsd_p95": double_p95,
        "contexts": len(contexts),
        "total_directions": int(sum(direction_counts)),
        "directions_per_context": direction_counts,
    }


def floor_ties(values: Sequence[float], floor: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or floor < 0:
        raise ValueError("tie-floor inputs must be finite one-dimensional values and floor >= 0")
    if not len(array):
        return array
    order = np.argsort(array, kind="stable")
    result = array.copy()
    group = [int(order[0])]
    group_start = float(array[order[0]])
    for raw_index in order[1:]:
        index = int(raw_index)
        value = float(array[index])
        # Bound the full span of a tie group. Comparing only adjacent gaps can
        # transitively chain values whose endpoints are much farther apart than
        # the registered numerical floor.
        if value - group_start <= floor:
            group.append(index)
        else:
            representative = float(np.mean(array[group]))
            result[group] = representative
            group = [index]
            group_start = value
    result[group] = float(np.mean(array[group]))
    return result


def tau_b_and_scale(
    response_distances: Sequence[float],
    rubric_distances: Sequence[float],
    *,
    response_tie_floor: float,
    rubric_tie_floor: float,
) -> tuple[float, float]:
    if len(response_distances) != len(rubric_distances) or len(response_distances) < 2:
        raise ValueError("Kendall tau-b requires equally sized distance vectors of length >= 2")
    left = floor_ties(response_distances, response_tie_floor)
    right = floor_ties(rubric_distances, rubric_tie_floor)
    result = kendalltau(left, right, variant="b", nan_policy="raise")
    tau = float(result.statistic)
    if not math.isfinite(tau):
        tau = 0.0
    return tau, 0.05 + 0.95 * max(0.0, tau)


@contextmanager
def exact_parameter_perturbation(
    parameters: Sequence[torch.nn.Parameter],
    direction: Sequence[torch.Tensor],
    epsilon: float,
) -> Iterator[None]:
    if len(parameters) != len(direction) or epsilon < 0:
        raise ValueError("parameter/direction mismatch or negative epsilon")
    originals = [parameter.detach().cpu().clone() for parameter in parameters]
    theta_norm_tensor = torch.sqrt(sum(
        parameter.detach().float().square().sum() for parameter in parameters
    ))
    theta_norm = float(theta_norm_tensor.item())
    direction_norm = math.sqrt(sum(
        float(value.detach().float().square().sum().item())
        for value in direction
    ))
    if not math.isfinite(theta_norm) or not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("perturbation norm is non-finite or zero")
    scale = epsilon * theta_norm / direction_norm
    try:
        with torch.no_grad():
            for parameter, value in zip(parameters, direction):
                parameter.add_(value.to(device=parameter.device, dtype=parameter.dtype), alpha=float(scale))
        yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original.to(device=parameter.device, dtype=parameter.dtype))
