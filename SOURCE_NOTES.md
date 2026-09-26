# Source notes

The following modules retain the experimental source verbatim:
`calibration/math.py`, `calibration/probe.py`, `calibration/scorer.py`,
`calibration/judge.py`, and `calibration/vllm_generate.py`.

The definitions in `mmpo.py`, `robust.py`, `calibration/adapters.py`, and
`calibration/schema.py` were selected from the experimental implementation;
imports were reduced to their standalone dependencies. The schema retains
the production scorer protocol identifier, including its retry version.
That identifier records provenance: the small standalone scorer does not
implement the production three-attempt/chunk-fallback retry scheduler.

The command-line wrappers, synthetic examples, tiny-model runner, and focused
tests are supplementary packaging. The single-device training wrapper uses
the same loss and shared-dual updates but replaces the production FSDP trainer;
its batching and learning-rate scheduling are not claimed to reproduce the
production trainer step for step.

External model checkpoints are downloaded separately and remain subject to
their own model licenses:

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [RubricARROW-8B-Judge](https://huggingface.co/OpenRubrics/RubricARROW-8B-Judge)

The rubric scoring prompt and Boolean criterion interface follow RubricARROW.
The scorer source included here is the experimental adaptation for exact
Boolean-branch probabilities and strict output schema checking.

No model weights, dataset records, personal server configurations, credentials,
or original repository history are included. The examples were authored only
for this supplementary package. No additional redistribution license is
asserted for external models or data.
