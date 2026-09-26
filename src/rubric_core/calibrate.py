"""Small file-based drivers around the original Fisher probe and exact scorer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .calibration.math import epsilon_correctness_gate, tau_b_and_scale
from .calibration.schema import FisherCalibrationConfig, RubricCriterion, FROZEN_RUBRIC_TIE_FLOOR
from .calibration.probe import calibrate_probe_context
from .calibration.judge import build_batched_judge_requests, score_generated_requests_exact, rubric_distances_from_batched_scores
from .io import read_jsonl, write_jsonl

PROBE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SCORER_REVISION = "a76e107c09ee6d599c51bddc1d799e834bdc5064"


def load_model(path, attention):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("The model calibration driver requires a CUDA GPU; use rubric-demo on CPU")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation=attention, device_map={"": 0}, low_cpu_mem_usage=True)
    model.eval()
    return model, tokenizer


def criteria_for(row):
    return tuple(RubricCriterion(
        criterion=item["criterion"], guidance=item["guidance"], anchors=tuple(item["anchors"]),
        importance=float(item["importance"]), direction=item.get("direction", "positive"),
    ) for item in row["criteria"])


def probe_main():
    parser = argparse.ArgumentParser(description="Sample Fisher directions and prepare rubric judge requests")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correctness", action="store_true", help="Evaluate 0, epsilon/2, epsilon, 2*epsilon")
    parser.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory for each calibration run")
    out.mkdir(parents=True, exist_ok=True)
    contexts = read_jsonl(args.input, unique_contexts=True)
    config = FisherCalibrationConfig(probe_revision=PROBE_REVISION, scorer_revision=SCORER_REVISION,
                                      epsilon=args.epsilon)
    model, tokenizer = load_model(args.model, args.attention)
    probes, requests = [], []
    for index, context in enumerate(contexts):
        criteria = criteria_for(context)
        seed = args.seed + index * 10_000
        retry = False
        try:
            def run(current_seed):
                return calibrate_probe_context(model, tokenizer, context["prompt_messages"], config,
                    epsilon=args.epsilon, seed=current_seed,
                    epsilon_schedule=(0., args.epsilon / 2, args.epsilon, 2 * args.epsilon) if args.correctness else None)
            try:
                result = run(seed)
            except ValueError as error:
                if "rank_below_min" not in str(error):
                    raise
                retry = True
                result = run(seed + 1_000_000)
            row = {"context_id": context["context_id"], "status": "probe_complete",
                   "rank_resampled_once": retry, **asdict(result)}
            requests.extend(build_batched_judge_requests(context["context_id"], context["prompt_messages"],
                                                         criteria, result.paired_responses))
        except (ValueError, RuntimeError) as error:
            row = {"context_id": context["context_id"], "status": "failed",
                   "rank_resampled_once": retry, "failure_reason": f"{type(error).__name__}: {error}"}
            torch.cuda.empty_cache()
        probes.append(row)
        write_jsonl(out / "probes.jsonl", probes)
        if requests:
            write_jsonl(out / "judge_requests.jsonl", requests)
        print(json.dumps({"context_id": context["context_id"], "status": row["status"]}), flush=True)
    report = {"config": config.canonical_dict(), "seed": args.seed, "attention": args.attention,
              "complete": sum(row["status"] == "probe_complete" for row in probes), "total": len(probes),
              "checkpoint_note": "Revision strings document the experimental checkpoints; local files are user-supplied."}
    valid = [row for row in probes if row["status"] == "probe_complete"]
    if args.correctness and valid:
        report["epsilon_correctness"] = epsilon_correctness_gate(valid[0]["epsilon_values"],
            [row["response_distances_by_epsilon"] for row in valid])
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(valid) != len(probes):
        raise SystemExit("Some probes failed; inspect probes.jsonl. No default s is assigned.")
    if args.correctness and not report["epsilon_correctness"]["passed"]:
        raise SystemExit("Epsilon correctness gate failed; do not promote this setting to a full run.")


def score_main():
    parser = argparse.ArgumentParser(description="Score saved judge generations using raw true/false probabilities")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Same contexts JSONL used by rubric-probe")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-tie-floor", type=float, required=True)
    parser.add_argument("--rubric-tie-floor", type=float, default=FROZEN_RUBRIC_TIE_FLOOR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0
           for value in (args.response_tie_floor, args.rubric_tie_floor)):
        raise ValueError("Tie floors must be finite and nonnegative")
    contexts = {row["context_id"]: row for row in read_jsonl(args.input, unique_contexts=True)}
    probes = read_jsonl(args.probes, unique_contexts=True)
    requests = read_jsonl(args.requests)
    generations = read_jsonl(args.generations)
    model, tokenizer = load_model(args.model, args.attention)
    scores = score_generated_requests_exact(model, tokenizer, requests, generations, batch_size=args.batch_size)
    output = []
    for probe in probes:
        context_id = probe["context_id"]
        if probe["status"] != "probe_complete":
            output.append({"context_id": context_id, "status": "failed", "failure_reason": probe.get("failure_reason")})
            continue
        try:
            local = [request for request in requests if request["context_id"] == context_id]
            rubric, records = rubric_distances_from_batched_scores(criteria_for(contexts[context_id]),
                probe["paired_responses"], local, scores)
            tau, scale = tau_b_and_scale(probe["response_distances"], rubric,
                response_tie_floor=args.response_tie_floor, rubric_tie_floor=args.rubric_tie_floor)
            output.append({"context_id": context_id, "status": "complete", "s": scale, "tau_b": tau,
                           "response_distances": probe["response_distances"], "rubric_distances": rubric,
                           "judge_records": records})
        except (ValueError, KeyError, AssertionError) as error:
            output.append({"context_id": context_id, "status": "failed", "failure_reason": str(error)})
    write_jsonl(args.output, output)
    print(json.dumps({"complete": sum(row["status"] == "complete" for row in output), "total": len(output)}))
    if any(row["status"] != "complete" for row in output):
        raise SystemExit("Some criteria could not be scored; inspect failures. No default s is assigned.")


if __name__ == "__main__":
    probe_main()
