"""Compute s from measured response distances and criterion score vectors."""
from __future__ import annotations

import argparse
import math
import numpy as np

from .calibration.math import tau_b_and_scale
from .calibration.schema import FROZEN_RUBRIC_TIE_FLOOR
from .io import read_jsonl, write_jsonl


def score_context(row, *, response_tie_floor, rubric_tie_floor):
    for value in (response_tie_floor, rubric_tie_floor):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Tie floors must be finite and nonnegative")
    response = np.asarray(row["response_distances"], dtype=float)
    if response.ndim != 1 or not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("response_distances must be a finite nonnegative vector")
    if "rubric_distances" in row:
        rubric = np.asarray(row["rubric_distances"], dtype=float)
    else:
        base = np.asarray(row["base_scores"], dtype=float)
        perturbed = np.asarray(row["perturbed_scores"], dtype=float)
        if base.shape != perturbed.shape or base.ndim != 2 or base.shape[0] != len(response):
            raise ValueError("For R=1, score arrays must have equal shape [directions, criteria]")
        if base.shape[1] == 0:
            raise ValueError("At least one criterion is required")
        if not np.isfinite(base).all() or not np.isfinite(perturbed).all():
            raise ValueError("Criterion scores must be finite")
        if (base < 0).any() or (base > 1).any() or (perturbed < 0).any() or (perturbed > 1).any():
            raise ValueError("Criterion scores must lie in [0,1]")
        # Same unweighted vector L2 used in the experiment's judge adapter.
        rubric = np.linalg.norm(base - perturbed, axis=-1)
    if rubric.shape != response.shape or not np.isfinite(rubric).all() or (rubric < 0).any():
        raise ValueError("rubric_distances must match the response vector")
    tau, scale = tau_b_and_scale(
        response, rubric, response_tie_floor=response_tie_floor,
        rubric_tie_floor=rubric_tie_floor,
    )
    return {
        "context_id": row["context_id"], "status": "complete",
        "response_distances": response.tolist(), "rubric_distances": rubric.tolist(),
        "directions_used": len(response), "tau_b": tau, "s": scale,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-tie-floor", type=float, required=True)
    parser.add_argument("--rubric-tie-floor", type=float, default=FROZEN_RUBRIC_TIE_FLOOR)
    args = parser.parse_args()
    rows = [score_context(row, response_tie_floor=args.response_tie_floor,
                          rubric_tie_floor=args.rubric_tie_floor)
            for row in read_jsonl(args.input, unique_contexts=True)]
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} context scores to {args.output}")


if __name__ == "__main__":
    main()
