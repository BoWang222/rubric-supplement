# Rubric-aware MMPO: supplementary core implementation

This repository contains the experimental algorithms for constructing the
context-dependent scale **s** and optimizing **MMPO**, including the KL and
binary Wasserstein variants. It includes small standalone runners, pinned
dependencies, synthetic examples, and tests.

## Install

Use Python 3.11 (the experimental version) or 3.12. From the extracted repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[models,test]'
```

The main environment pins PyTorch 2.5.1 and Transformers 4.52.4. The CPU examples
below need no model downloads or credentials. For Linux/CUDA and the separate
vLLM environment, see [GPU instructions](docs/gpu.md).

## Quick start on CPU

```bash
# Fisher/SVD, distance, s, and MMPO forward/backward demonstration
rubric-demo

# Compute s from saved distances or criterion score vectors
rubric-compute-s --input examples/scored_probes.jsonl \
  --output outputs/toy-s.jsonl --response-tie-floor 0 --rubric-tie-floor 0

# Run three training steps with a tiny randomly initialized local model
python -m rubric_core.toy --backend kl --output outputs/toy-kl

# Check numerical identities, robust solvers, and actual model updates
python -m pytest
```

All files in `examples/` are **synthetic illustrations, not experimental data**.
The zero tie floors and toy learning rate are only for these demonstrations.
Training outputs must use a new or empty directory. The toy command also accepts
`--backend none` and `--backend wasserstein_w1`.

## What is included

| File | Role |
| --- | --- |
| `src/rubric_core/calibration/probe.py` | Response gradients, centered Fisher directions, parameter perturbation, response distances |
| `src/rubric_core/calibration/math.py` | Gram/SVD operations, full-vocabulary JSD, tie grouping, Kendall tau-b, s |
| `src/rubric_core/calibration/scorer.py` and `judge.py` | Actual rubric judge prompt, exact Boolean-branch scoring, criterion-vector distances |
| `src/rubric_core/mmpo.py` | MMPO soft-target loss and policy/reference log-ratio |
| `src/rubric_core/robust.py` | Binary KL/W1 adversaries and shared dual controller |
| `src/rubric_core/objective.py` | Nominal or robust MMPO integration |
| `src/rubric_core/train.py` | Minimal single-device training with cached reference log probabilities |
| `configs/experiment.json` | Frozen algorithm settings and production training settings |

### Construction of s

The probe samples 16 responses and differentiates their mean token log
probabilities with respect to the final **two** transformer blocks and final
RMSNorm. Centered response gradients define up to 8 leading directions, with a
minimum rank of 4. Each direction applies a positive relative perturbation of
magnitude `epsilon * ||theta||`; the original parameters are restored exactly.

For each direction, the response distance is the average full-vocabulary token
JSD on fixed base-response prefixes. The rubric distance is the **unweighted L2**
distance between criterion score vectors for paired generations (`R=1`). Each
criterion score is `(1 + p_true - p_false) / 2`, using raw model probabilities,
without renormalizing the two Boolean branches. The criterion `importance`
field does not weight this distance.

After applying the measurement tie floors to each distance vector:

```text
s = 0.05 + 0.95 * max(0, Kendall_tau_b(response_distances, rubric_distances))
```

Undefined correlation, including constant tied vectors, is mapped to zero and
therefore `s=0.05`. Tie groups have full span no greater than the floor; adjacent
small gaps do not create transitive chains. Failed probes or invalid judge
outputs are reported as failures and are not assigned a default s.

### MMPO and robust optimization

With prepared normalized preference margin `m`:

```text
p0 = clip(sigmoid(gamma * m), 1e-6, 1 - 1e-6)
z  = beta * ((log_pi_chosen - log_ref_chosen)
          - (log_pi_rejected - log_ref_rejected))
L  = p0 * softplus(-z) + (1 - p0) * softplus(z)
```

Completion log probabilities are **summed**, including EOS and excluding prompt
and padding tokens. Defaults are `beta=0.01` and `gamma=2.2`.

Robust MMPO replaces `p0` with an adversarial Bernoulli preference probability.
The mean transport cost is `s * KL(q || p0) / (-log(1e-6))` for KL, or
`s * abs(q - p0)` for binary W1 (unit ground cost). Training uses the experimental
**shared positive dual variable**, updated with Adam at learning rate 0.01;
the adversarial probabilities are detached for the policy gradient. The radius
is enforced through this stochastic dual update, not solved exactly on each
minibatch. Exact finite-batch solvers are also included for verification.
`rho=0` reduces to nominal MMPO; `--s-mode uniform` sets all scales to one.

## Scope and validation

The core functions were extracted from the experimental implementation. The
standalone drivers are intentionally small: production FSDP orchestration,
dataset-specific preprocessing, full evaluation suites, model weights, and
experiment outputs are not distributed here. The scorer runner retains the
exact scoring rule but omits the production retry/cache scheduler; invalid
generations fail explicitly. This is a core implementation release, not a
one-command reproduction of every paper table.

CPU validation covers numerical Fisher identities, parameter restoration,
KL/W1 solver checks, zero-radius loss/gradient agreement, prompt/padding masks,
and real tiny-model training/saving with all three backends. The full 8B CUDA
probe and vLLM pipeline require suitable hardware and are not part of that CPU
validation. See [GPU instructions](docs/gpu.md) for data formats and launch
commands, and [source notes](SOURCE_NOTES.md) for extraction details.
