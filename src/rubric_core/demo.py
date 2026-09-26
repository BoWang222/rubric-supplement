"""Offline CPU illustration of the complete mathematical core."""
from __future__ import annotations

import json
import torch

from .calibration.math import adaptive_gram_svd, direction_coefficients, full_softmax_jsd
from .compute_s import score_context
from .objective import MMPOObjective


def main():
    torch.manual_seed(42)
    gradients = torch.randn(16, 24, dtype=torch.float64)
    singular, vectors, rank = adaptive_gram_svd(gradients @ gradients.T, max_k=8, min_k=4)
    directions = (gradients - gradients.mean(0)).T @ direction_coefficients(singular, vectors)
    assert torch.allclose(directions.T @ directions, torch.eye(8, dtype=torch.float64), atol=1e-8)
    logits = torch.randn(5, 11)
    changes = torch.randn_like(logits)
    distances = [full_softmax_jsd(logits, logits + step * changes) for step in torch.linspace(.01, .08, 8)]
    scored = score_context({
        "context_id": "synthetic-demo", "response_distances": distances,
        "base_scores": [[0.0] * 4 for _ in distances],
        "perturbed_scores": [[float(value)] * 4 for value in torch.linspace(.1, .8, 8)],
    }, response_tie_floor=0.0, rubric_tie_floor=0.0)
    output = {"data": "synthetic illustration, not experimental results", "rank": rank,
              "directions": len(singular), "tau_b": scored["tau_b"], "s": scored["s"]}
    for backend in ("none", "kl", "wasserstein_w1"):
        chosen = torch.tensor([.4, -.2], requires_grad=True)
        zeros = torch.zeros(2)
        objective = MMPOObjective(backend=backend, rho=0.02)
        loss, diagnostics = objective(chosen, zeros, zeros, zeros,
                                     torch.tensor([.7, .4]), torch.tensor([scored["s"], .3]))
        loss.backward()
        assert torch.isfinite(chosen.grad).all()
        output[backend] = {"loss": float(loss.detach()), "gradient": chosen.grad.tolist(), **diagnostics}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
