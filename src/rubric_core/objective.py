"""MMPO integration preserving the experiment's shared dual update."""
from __future__ import annotations

import math
import torch

from .mmpo import mmpo_loss_from_p0, preference_logits
from .robust import DistributedDualController, robust_preference_loss


class MMPOObjective:
    def __init__(self, *, beta=0.01, gamma=2.2, backend="none", rho=0.0,
                 s_mode="context", device="cpu"):
        if backend not in ("none", "kl", "wasserstein_w1"):
            raise ValueError("Unknown robust backend")
        if s_mode not in ("context", "uniform"):
            raise ValueError("s_mode must be context or uniform")
        if not all(math.isfinite(x) for x in (rho, beta, gamma)) or not 0 <= rho <= 1 or beta <= 0 or gamma <= 0:
            raise ValueError("Invalid MMPO hyperparameters")
        self.beta, self.gamma, self.backend, self.rho = beta, gamma, backend, rho
        self.s_mode = s_mode
        self.controller = DistributedDualController().to(device) if backend != "none" else None

    def __call__(self, chosen, rejected, ref_chosen, ref_rejected, margin, rubric_s,
                 *, update_dual=True):
        logits, _, _ = preference_logits(chosen, rejected, ref_chosen, ref_rejected, self.beta)
        if not torch.isfinite(margin).all() or not ((margin > 0) & (margin <= 1)).all():
            raise ValueError("Prepared margin_normalized must be in (0,1]")
        p0 = torch.sigmoid(self.gamma * margin).clamp(1e-6, 1.0 - 1e-6)
        if self.backend == "none":
            return mmpo_loss_from_p0(logits, p0).mean(), {"p0_mean": float(p0.detach().mean())}
        scale = torch.ones_like(p0) if self.s_mode == "uniform" else rubric_s
        result = robust_preference_loss(
            logits, p0, scale, backend=self.backend, rho=self.rho,
            dual_eta=self.controller.eta.detach() if self.rho > 0 else None,
        )
        if update_dual and self.rho > 0:
            self.controller.update(torch.tensor(result.diagnostics["distance"], device=logits.device), self.rho)
        return result.loss, result.diagnostics
