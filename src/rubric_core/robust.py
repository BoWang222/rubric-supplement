from __future__ import annotations
import math
import copy
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn.functional as F
ROBUST_BACKENDS = ("none", "kl", "wasserstein_w1")
LABEL_KL_NORMALIZER = -math.log(1e-6)
KL_NORMALIZER = LABEL_KL_NORMALIZER

@dataclass(frozen=True)
class RobustOutput:
    loss: torch.Tensor
    adversarial_distribution: torch.Tensor
    diagnostics: dict[str, float]


def _validate_binary_inputs(
    nominal_distribution: torch.Tensor,
    branch_losses: torch.Tensor,
    rubric_s: torch.Tensor,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = nominal_distribution.float()
    losses = branch_losses.float()
    s = rubric_s.float()
    if p.ndim != 1 or losses.shape != (p.numel(), 2) or s.shape != p.shape:
        raise ValueError("binary DRO expects p:[B], branch_losses:[B,2], s:[B]")
    if not torch.isfinite(p).all() or not torch.isfinite(losses).all() or not torch.isfinite(s).all():
        raise ValueError("DRO inputs must be finite")
    if not ((p >= 1e-6) & (p <= 1.0 - 1e-6)).all():
        raise ValueError("nominal preference probability must be in [1e-6,1-1e-6]")
    if not ((s >= 0.05) & (s <= 1.0)).all():
        raise ValueError("rubric s must be in [0.05,1]")
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and in [0,1]")
    return p, losses, s


def binary_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    q = q.clamp(1e-12, 1.0 - 1e-12)
    p = p.clamp(1e-12, 1.0 - 1e-12)
    return q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())


def _global_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().float().mean()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def _weighted_objective(q: torch.Tensor, branch_losses: torch.Tensor) -> torch.Tensor:
    return q * branch_losses[:, 0] + (1.0 - q) * branch_losses[:, 1]


class BinaryKLDRO:
    name = "kl"

    def __init__(self, *, iterations: int = 80):
        self.iterations = int(iterations)

    def solve(self, nominal_distribution, branch_losses, rubric_s, rho) -> RobustOutput:
        p, losses, s = _validate_binary_inputs(nominal_distribution, branch_losses, rubric_s, rho)
        nominal_loss = _weighted_objective(p, losses)
        if rho == 0.0:
            return RobustOutput(
                nominal_loss.mean().to(branch_losses.dtype),
                torch.stack((p, 1.0 - p), dim=-1).to(branch_losses.dtype),
                {"rho": 0.0, "distance": 0.0, "dual_eta": 0.0},
            )
        delta = losses[:, 0] - losses[:, 1]
        worst_q = (delta >= 0).float().clamp(1e-6, 1.0 - 1e-6)
        worst_distance = _global_mean(s * binary_kl(worst_q, p) / LABEL_KL_NORMALIZER)
        if float(worst_distance) <= rho:
            q, eta = worst_q, 0.0
        else:
            low = torch.tensor(0.0, device=p.device)
            high = torch.tensor(1.0, device=p.device)

            def candidate(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                safe = value.clamp_min(1e-12)
                logits = torch.logit(p) + delta * LABEL_KL_NORMALIZER / (safe * s)
                candidate_q = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
                distance = _global_mean(s * binary_kl(candidate_q, p) / LABEL_KL_NORMALIZER)
                return candidate_q, distance

            _, high_distance = candidate(high)
            while float(high_distance) > rho:
                high = high * 2.0
                _, high_distance = candidate(high)
                if float(high) > 1e12:
                    raise RuntimeError("KL dual bracketing failed")
            q = p
            for _ in range(self.iterations):
                middle = (low + high) / 2.0
                candidate_q, distance = candidate(middle)
                if float(distance) > rho:
                    low = middle
                else:
                    high, q = middle, candidate_q
            eta = float(high.item())
        distance = _global_mean(s * binary_kl(q, p) / LABEL_KL_NORMALIZER)
        objective = _weighted_objective(q.detach(), losses)
        return RobustOutput(
            objective.mean().to(branch_losses.dtype),
            torch.stack((q, 1.0 - q), dim=-1).to(branch_losses.dtype),
            {
                "rho": float(rho), "distance": float(distance.item()),
                "constraint_slack": float(rho - distance.item()), "dual_eta": eta,
                "q_mean": float(_global_mean(q).item()),
            },
        )

    def solve_with_dual(
        self, nominal_distribution, branch_losses, rubric_s, rho, dual_eta: torch.Tensor | float
    ) -> RobustOutput:
        """Evaluate the global-dual inner solution without re-solving eta per batch."""
        p, losses, s = _validate_binary_inputs(nominal_distribution, branch_losses, rubric_s, rho)
        if rho == 0.0:
            return self.solve(p, losses, s, 0.0)
        eta = torch.as_tensor(dual_eta, device=p.device, dtype=torch.float32).clamp_min(1e-8)
        delta = losses[:, 0] - losses[:, 1]
        q = torch.sigmoid(torch.logit(p) + delta * LABEL_KL_NORMALIZER / (eta * s)).clamp(
            1e-6, 1.0 - 1e-6
        )
        distance = _global_mean(s * binary_kl(q, p) / LABEL_KL_NORMALIZER)
        objective = _weighted_objective(q.detach(), losses)
        return RobustOutput(
            objective.mean().to(branch_losses.dtype),
            torch.stack((q, 1.0 - q), dim=-1).to(branch_losses.dtype),
            {
                "rho": float(rho), "distance": float(distance.item()),
                "constraint_slack": float(rho - distance.item()),
                "dual_eta": float(eta.detach().item()), "q_mean": float(_global_mean(q).item()),
            },
        )


class BinaryWassersteinDRO:
    """Exact binary W1 solver for c(0,1)=1, so W1=|q-p|."""

    name = "wasserstein_w1"

    def solve(self, nominal_distribution, branch_losses, rubric_s, rho) -> RobustOutput:
        p, losses, s = _validate_binary_inputs(nominal_distribution, branch_losses, rubric_s, rho)
        if rho == 0.0:
            nominal = _weighted_objective(p, losses)
            return RobustOutput(
                nominal.mean().to(branch_losses.dtype),
                torch.stack((p, 1.0 - p), dim=-1).to(branch_losses.dtype),
                {"rho": 0.0, "distance": 0.0, "dual_eta": 0.0},
            )
        delta = losses[:, 0] - losses[:, 1]
        target = torch.where(delta >= 0, torch.ones_like(p), torch.zeros_like(p))
        capacity = (target - p).abs()
        budget = min(float(rho) * p.numel(), float((s * capacity).sum().item()))
        moved = torch.zeros_like(p)
        order = torch.argsort(delta.abs() / s, descending=True)
        remaining, dual_eta = budget, 0.0
        for raw_index in order.tolist():
            cost_per_unit = float(s[raw_index].item())
            available = float(capacity[raw_index].item())
            amount = min(available, remaining / cost_per_unit)
            moved[raw_index] = amount
            remaining -= amount * cost_per_unit
            dual_eta = float(delta[raw_index].abs().item() / cost_per_unit)
            if remaining <= 1e-12:
                break
        q = p + torch.sign(target - p) * moved
        distance = _global_mean(s * (q - p).abs())
        objective = _weighted_objective(q.detach(), losses)
        return RobustOutput(
            objective.mean().to(branch_losses.dtype),
            torch.stack((q, 1.0 - q), dim=-1).to(branch_losses.dtype),
            {
                "rho": float(rho), "distance": float(distance.item()),
                "constraint_slack": float(rho - distance.item()), "dual_eta": dual_eta,
                "q_mean": float(_global_mean(q).item()),
            },
        )

    def solve_with_dual(
        self, nominal_distribution, branch_losses, rubric_s, rho, dual_eta: torch.Tensor | float
    ) -> RobustOutput:
        p, losses, s = _validate_binary_inputs(nominal_distribution, branch_losses, rubric_s, rho)
        if rho == 0.0:
            return self.solve(p, losses, s, 0.0)
        eta = torch.as_tensor(dual_eta, device=p.device, dtype=torch.float32).clamp_min(1e-8)
        delta = losses[:, 0] - losses[:, 1]
        target = torch.where(delta >= 0, torch.ones_like(p), torch.zeros_like(p))
        move = delta.abs() > eta * s
        q = torch.where(move, target, p)
        distance = _global_mean(s * (q - p).abs())
        objective = _weighted_objective(q.detach(), losses)
        return RobustOutput(
            objective.mean().to(branch_losses.dtype),
            torch.stack((q, 1.0 - q), dim=-1).to(branch_losses.dtype),
            {
                "rho": float(rho), "distance": float(distance.item()),
                "constraint_slack": float(rho - distance.item()),
                "dual_eta": float(eta.detach().item()), "q_mean": float(_global_mean(q).item()),
            },
        )


def robust_preference_loss(
    logits: torch.Tensor,
    nominal_preference_probability: torch.Tensor,
    rubric_s: torch.Tensor,
    *, backend: str, rho: float, dual_eta: torch.Tensor | float | None = None,
) -> RobustOutput:
    if backend not in ROBUST_BACKENDS:
        raise ValueError(f"unknown robust backend {backend!r}")
    branch_losses = torch.stack((F.softplus(-logits), F.softplus(logits)), dim=-1)
    if backend == "none":
        p, losses, _ = _validate_binary_inputs(
            nominal_preference_probability, branch_losses, rubric_s, 0.0
        )
        nominal = _weighted_objective(p, losses)
        return RobustOutput(
            nominal.mean(), torch.stack((p, 1.0 - p), dim=-1),
            {"rho": 0.0, "distance": 0.0, "dual_eta": 0.0},
        )
    solver = BinaryKLDRO() if backend == "kl" else BinaryWassersteinDRO()
    if dual_eta is not None:
        return solver.solve_with_dual(
            nominal_preference_probability, branch_losses, rubric_s, rho, dual_eta
        )
    return solver.solve(nominal_preference_probability, branch_losses, rubric_s, rho)


class DistributedDualController(torch.nn.Module):
    """Checkpointable FP32 positive scalar controller, separate from policy weights."""

    def __init__(self, initial_eta: float = 1.0, learning_rate: float = 1e-2):
        super().__init__()
        if initial_eta <= 0:
            raise ValueError("initial eta must be positive")
        self.raw_eta = torch.nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_eta)), dtype=torch.float32)
        )
        self.optimizer = torch.optim.Adam([self.raw_eta], lr=learning_rate)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta) + 1e-8

    def update(self, observed_distance: torch.Tensor, rho: float) -> float:
        distance = observed_distance.detach().float().mean()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(distance, op=dist.ReduceOp.SUM)
            distance /= dist.get_world_size()
        self.optimizer.zero_grad(set_to_none=True)
        loss = -self.eta * (distance - float(rho))
        loss.backward()
        self.optimizer.step()
        self.updates.add_(1)
        return float(self.eta.detach().item())

    def checkpoint_state(self) -> dict[str, object]:
        return copy.deepcopy({"module": self.state_dict(), "optimizer": self.optimizer.state_dict()})

    def load_checkpoint_state(self, state: dict[str, object]) -> None:
        self.load_state_dict(state["module"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
