# GPU setup and real-model execution

Run commands from the repository root. Use Linux, Python 3.11, an NVIDIA GPU
supporting BF16, and a CUDA-compatible driver. Full-parameter 8B training and
the gradient-bank probe need substantial GPU and host RAM; a small inference
GPU is insufficient. The original distributed launcher is outside this release.

## Environments and checkpoints

The main environment is separate from vLLM because the pinned vLLM release has
different PyTorch requirements. For the main environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[models,test]'
python -m pytest
```

The standalone commands default to PyTorch SDPA. The original runs used
FlashAttention 2; `--attention flash_attention_2` requires a compatible separate
installation (the experimental version was `flash-attn==2.7.4.post1`). The
portable package pins NumPy 1.26.4 and SciPy 1.15.3 for the numerical core; it
does not freeze every dependency of the original production environment.

Download the public checkpoints at the recorded revisions:

```bash
huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir models/qwen3-8b
huggingface-cli download OpenRubrics/RubricARROW-8B-Judge \
  --revision a76e107c09ee6d599c51bddc1d799e834bdc5064 \
  --local-dir models/rubricarrow

python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install -r requirements-vllm.txt
.venv-vllm/bin/python -m pip install --no-deps -e .
```

Model loaders use local files and disable remote code. Revision strings in the
run metadata document the experimental checkpoints; the driver does not verify
the contents of an arbitrary local model directory.

## Measure s

Prepare a JSONL file with one row per unique `context_id`. Each row contains
`prompt_messages` and `criteria`. Each criterion has `criterion`, `guidance`,
nonempty `anchors`, `importance` in `(0,1]`, and optional `direction` (`positive`
or `negative`). See `examples/contexts.jsonl` for the schema; replace these toy
contexts with your own data and generated or supplied rubrics.

```bash
rubric-probe --model models/qwen3-8b --input examples/contexts.jsonl \
  --output-dir outputs/probe --epsilon 3e-5

.venv-vllm/bin/python -m rubric_core.calibration.vllm_generate \
  --model models/rubricarrow \
  --requests outputs/probe/judge_requests.jsonl \
  --output outputs/judge-generations.jsonl \
  --metrics outputs/judge-metrics.json

rubric-score --model models/rubricarrow --input examples/contexts.jsonl \
  --probes outputs/probe/probes.jsonl \
  --requests outputs/probe/judge_requests.jsonl \
  --generations outputs/judge-generations.jsonl \
  --output outputs/s.jsonl --response-tie-floor 1e-8 \
  --rubric-tie-floor 0.7070668974154677
```

These commands run sequentially so the probe and judge do not coexist on the
GPU. vLLM generates the judge output; the main environment then performs exact
teacher-forced scoring of its Boolean branches. Judge decoding uses seed 42,
temperature 1.0, and top-p 0.95; rubric chunks contain at most 8 criteria.

The epsilon and tie floors above are recorded RubricBench settings, not
universal defaults. For a new model or setup, run a separate calibration pilot:
add `--correctness` to `rubric-probe` and use a new output directory. This checks
response JSD at zero, half, nominal, and double epsilon, and writes the numerical
noise floor and gate outcome to `run.json`. Freeze the chosen epsilon and
response floor before the main run. A different judge/decoding setup also
requires a separate duplicate-decoding pilot for the rubric noise floor;
that pilot scheduler is not included here.

The probe retries insufficient Fisher rank once with a new seed. Scorer schema
or token-alignment errors are explicit failures. The standalone scorer does
not automatically retry malformed outputs; inspect failed contexts and rerun
their generation/scoring before using the resulting s values. Do not substitute
`s=0.05` for failed measurements. Successfully measured constant/tied distances
can legitimately yield `s=0.05`.

## Train MMPO

Prepare JSONL pairs with `context_id`, `prompt_messages`, string `chosen` and
`rejected`, and `margin_normalized` in `(0,1]`. Supply measured `rubric_s` directly
or join the measured file with `--scores`. Multiple preference pairs may share
a context. The initial checkpoint is also the frozen reference; its completion
log probabilities are cached before training begins.

Dataset preprocessing is external to this package. In the experiment, orient
pairs toward the higher score, exclude ties/invalid records, and normalize the
positive score gap by the **training-only** 95th percentile using NumPy's linear
quantile, then clip to `[0,1]`. Apply the experiment's near-tie cutoff of 0.05
and reuse the training normalization for validation/test. Keep the same retained
contexts for nominal, uniform-s, and context-s comparisons.

```bash
rubric-train-mmpo --model models/qwen3-8b --input examples/pairs.jsonl \
  --scores outputs/s.jsonl --output outputs/mmpo-kl \
  --backend kl --rho 0.02 --s-mode context \
  --batch-size 1 --accumulation-steps 64 --epochs 1 \
  --lr 1e-6 --beta 0.01 --gamma 2.2 --gradient-checkpointing
```

Replace the example input with prepared training pairs for meaningful training.
`rho=0.02` is illustrative, not a claimed selected experimental radius. Select
the radius and learning rate on held-out validation. For nominal MMPO, use
`--backend none --rho 0`; for binary W1, use `--backend wasserstein_w1`.
Continue passing the same `--scores` file to apply the same eligibility filter.

The driver saves `model/`, `reference_cache.jsonl`, `training.jsonl`,
`training_state.pt` (optimizer, scheduler, and shared dual state), and `run.json`.
The shared dual is updated each microbatch. The driver does not implement
distributed FSDP, automatic checkpoint resume, or final benchmark evaluation.
Full-parameter Adam states may exceed one GPU's memory for an 8B model; integrate
`MMPOObjective` into an existing distributed trainer when needed.
