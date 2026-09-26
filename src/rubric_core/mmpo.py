from __future__ import annotations
import torch
import torch.nn.functional as F

def mmpo_loss_from_p0(z: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(p0).all() or not ((p0 >= 0) & (p0 <= 1)).all():
        raise ValueError("MMPO p0 must be finite and in [0,1]")
    return p0 * F.softplus(-z) + (1.0 - p0) * F.softplus(z)


def preference_logits(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return chosen_rewards - rejected_rewards, chosen_rewards.detach(), rejected_rewards.detach()
