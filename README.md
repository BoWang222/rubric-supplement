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
