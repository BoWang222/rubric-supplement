import itertools
import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rubric_core.calibration.math import (adaptive_gram_svd, direction_coefficients,
    epsilon_correctness_gate, exact_parameter_perturbation, floor_ties,
    full_softmax_jsd, tau_b_and_scale)
from rubric_core.calibration.probe import _centered_gram_from_gradient_banks, _direction_from_gradient_banks
from rubric_core.compute_s import score_context
from rubric_core.mmpo import mmpo_loss_from_p0
from rubric_core.objective import MMPOObjective
from rubric_core.robust import BinaryKLDRO, BinaryWassersteinDRO, binary_kl, LABEL_KL_NORMALIZER, DistributedDualController


def test_fisher_modes_match_explicit_svd():
    torch.manual_seed(9)
    gradients = torch.randn(16, 12, dtype=torch.float64)
    singular, vectors, rank = adaptive_gram_svd(gradients @ gradients.T, max_k=8, min_k=4)
    centered = gradients - gradients.mean(0)
    torch.testing.assert_close(singular, torch.linalg.svdvals(centered)[:8], atol=1e-10, rtol=1e-10)
    directions = centered.T @ direction_coefficients(singular, vectors)
    torch.testing.assert_close(directions.T @ directions, torch.eye(8, dtype=torch.float64))
    assert rank == 12


def test_low_rank_is_rejected_instead_of_filling_directions():
    gradients = torch.randn(16, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="rank_below_min"):
        adaptive_gram_svd(gradients @ gradients.T, max_k=8, min_k=4)


def test_cpu_gradient_bank_matches_explicit_centering():
    torch.manual_seed(3)
    bank = torch.randn(16, 30)
    gram = _centered_gram_from_gradient_banks([bank])
    centered = bank - bank.mean(0)
    torch.testing.assert_close(gram, (centered @ centered.T).double())
    singular, vectors, _ = adaptive_gram_svd(gram, max_k=8, min_k=4)
    coefficients = direction_coefficients(singular, vectors)[:, 0]
    parameter = torch.nn.Parameter(torch.zeros(5, 6))
    direction = _direction_from_gradient_banks([bank], [parameter], coefficients)[0]
    torch.testing.assert_close(direction.flatten(), (bank.double().T @ coefficients).float(), atol=2e-6, rtol=2e-6)


def test_perturbation_uses_relative_norm_and_restores_on_exception():
    parameter = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    original = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="intentional"):
        with exact_parameter_perturbation([parameter], [torch.tensor([2., 0., 1.])], .01):
            assert torch.linalg.vector_norm(parameter - original).item() == pytest.approx(.01 * original.norm().item(), rel=1e-5)
            raise RuntimeError("intentional")
    assert torch.equal(parameter, original)


def test_jsd_zero_symmetry_and_bound():
    a, b = torch.randn(7, 19), torch.randn(7, 19)
    assert full_softmax_jsd(a, a) == 0
    assert full_softmax_jsd(a, b) == pytest.approx(full_softmax_jsd(b, a))
    assert 0 <= full_softmax_jsd(a, b) <= np.log(2)


def test_noise_ties_do_not_chain_arbitrarily():
    tied = floor_ties([0., .6, 1.2], .7)
    assert tied[0] == tied[1] and tied[1] != tied[2]
    assert tau_b_and_scale([1, 2, 3], [3, 2, 1], response_tie_floor=0, rubric_tie_floor=0) == (-1., .05)
    assert tau_b_and_scale([1, 2, 3], [1, 2, 3], response_tie_floor=0, rubric_tie_floor=0) == (1., 1.)
    assert tau_b_and_scale([1, 1, 1], [1, 2, 3], response_tie_floor=0, rubric_tie_floor=0) == (0., .05)


def test_rubric_l2_uses_all_criteria_without_importance_weights():
    row = {"context_id": "test", "response_distances": [1, 2, 3, 4],
           "base_scores": [[0, 0]] * 4, "perturbed_scores": [[0, .2], [.3, .4], [.6, .8], [1, 1]]}
    result = score_context(row, response_tie_floor=0, rubric_tie_floor=0)
    assert result["rubric_distances"] == pytest.approx([.2, .5, 1., 2 ** .5])
    assert result["s"] == 1
    row["base_scores"][0] = [float("nan"), 0]
    with pytest.raises(ValueError, match="finite"):
        score_context(row, response_tie_floor=0, rubric_tie_floor=0)


@pytest.mark.parametrize("solver", [BinaryKLDRO(), BinaryWassersteinDRO()])
def test_binary_dro_agrees_with_exhaustive_small_grid(solver):
    p = torch.tensor([.7, .4])
    losses = torch.tensor([[.1, 1.1], [1.4, .2]])
    scale = torch.tensor([.6, 1.])
    rho = .015
    result = solver.solve(p, losses, scale, rho)
    grid = torch.linspace(1e-5, 1 - 1e-5, 501)
    q = torch.cartesian_prod(grid, grid)
    distance = ((scale * binary_kl(q, p) / LABEL_KL_NORMALIZER).mean(-1)
                if solver.name == "kl" else (scale * (q - p).abs()).mean(-1))
    values = (q * losses[:, 0] + (1 - q) * losses[:, 1]).mean(-1)
    exact_grid = values[distance <= rho].max()
    assert result.diagnostics["distance"] <= rho + 2e-6
    assert float(result.loss) >= float(exact_grid) - 2e-5
    assert float(result.loss) <= float(exact_grid) + .004


@pytest.mark.parametrize("backend", ["none", "kl", "wasserstein_w1"])
def test_zero_radius_recovers_mmpo_loss_and_gradient(backend):
    chosen = torch.tensor([.2, -.7, 1.3], requires_grad=True)
    zero = torch.zeros_like(chosen)
    margin, scale = torch.tensor([.8, .3, .6]), torch.tensor([.2, .7, 1.])
    objective = MMPOObjective(beta=.01, backend=backend, rho=0)
    loss, _ = objective(chosen, zero, zero, zero, margin, scale)
    expected = mmpo_loss_from_p0(.01 * chosen, torch.sigmoid(2.2 * margin)).mean()
    torch.testing.assert_close(loss, expected)
    g1 = torch.autograd.grad(loss, chosen, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, chosen)[0]
    torch.testing.assert_close(g1, g2)


def test_nominal_matches_binary_cross_entropy():
    logits = torch.tensor([-50., 0., 50.], requires_grad=True)
    target = torch.tensor([.6, .7, .8])
    torch.testing.assert_close(mmpo_loss_from_p0(logits, target), F.binary_cross_entropy_with_logits(logits, target, reduction="none"))


def test_shared_dual_state_round_trip():
    first = DistributedDualController(initial_eta=.7, learning_rate=.02)
    first.update(torch.tensor(.03), .02)
    state = first.checkpoint_state()
    second = DistributedDualController(initial_eta=1.2, learning_rate=.02)
    second.load_checkpoint_state(state)
    assert first.update(torch.tensor(.015), .02) == second.update(torch.tensor(.015), .02)
