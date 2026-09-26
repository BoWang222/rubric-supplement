from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Literal
from rubric_core.io import canonical_json
Direction = Literal["positive", "negative"]
LOCAL_EXACT_SCORER_PROTOCOL = (
    "rubricarrow_vllm_batch_generate_hf_exact_boolean_branches_exact_schema_retry_3_1_v8"
)
SCORER_SEED = 42
SCORER_CRITERION_CHUNK_SIZE = 8
# Registered from the independent duplicate-decoding pilot.  The production
# R=1 estimator consumes this frozen measurement-noise floor; it must not try
# to estimate a second stochastic replicate inside every Fisher direction.
FROZEN_RUBRIC_TIE_FLOOR = 0.7070668974154677




@dataclass(frozen=True)
class RubricCriterion:
    criterion: str
    guidance: str
    anchors: tuple[str, ...]
    importance: float
    direction: Direction = "positive"

    def __post_init__(self) -> None:
        if not self.criterion.strip() or not self.guidance.strip():
            raise ValueError("criterion and guidance must be non-empty")
        if not self.anchors or any(not value.strip() for value in self.anchors):
            raise ValueError("anchors must contain non-empty descriptions")
        if not 0.0 < float(self.importance) <= 1.0:
            raise ValueError("criterion importance must be in (0,1]")
        if self.direction not in ("positive", "negative"):
            raise ValueError(f"unsupported criterion direction {self.direction!r}")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "anchors": list(self.anchors),
            "criterion": self.criterion.strip(),
            "direction": self.direction,
            "guidance": self.guidance.strip(),
            "importance": float(self.importance),
        }


@dataclass(frozen=True)
class FisherCalibrationConfig:
    probe_revision: str
    scorer_revision: str
    scorer_protocol: str = LOCAL_EXACT_SCORER_PROTOCOL
    scorer_seed: int = SCORER_SEED
    scorer_criterion_chunk_size: int = SCORER_CRITERION_CHUNK_SIZE
    final_blocks: int = 2
    # Public method notation: M is the fixed initial response bank, K_max is
    # the direction cap, and each context uses k_i=min(K_max, numerical_rank).
    responses_m: int = 16
    directions_k: int = 8
    min_directions_k: int = 4
    paired_generations_r: int = 1
    epsilon: float | None = None
    epsilon_grid: tuple[float, ...] = (
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
    )
    decoding_temperature: float = 2.5
    decoding_top_p: float = 1.0
    max_new_tokens: int = 128
    top_logprobs: int = 10

    def __post_init__(self) -> None:
        if self.final_blocks != 2:
            raise ValueError("the registered probe subspace is frozen to final_blocks=2")
        if self.responses_m < 2 or not 0 < self.directions_k < self.responses_m:
            raise ValueError("directions_k must be positive and smaller than responses_m")
        if not 2 <= self.min_directions_k <= self.directions_k:
            raise ValueError("min_directions_k must be in [2, directions_k]")
        if self.paired_generations_r != 1:
            raise ValueError("paired_generations_r is frozen to R=1")
        if self.decoding_temperature != 2.5 or self.decoding_top_p != 1.0:
            raise ValueError("response decoding is frozen to temperature=2.5, top_p=1.0")
        if self.max_new_tokens != 128:
            raise ValueError("response max_new_tokens is frozen to 128")
        if self.top_logprobs != 10:
            raise ValueError("RubricARROW contract freezes top_logprobs=10")
        if self.scorer_protocol != LOCAL_EXACT_SCORER_PROTOCOL:
            raise ValueError("unsupported local scorer protocol")
        if self.scorer_seed != SCORER_SEED:
            raise ValueError(f"scorer_seed is frozen to {SCORER_SEED}")
        if self.scorer_criterion_chunk_size != SCORER_CRITERION_CHUNK_SIZE:
            raise ValueError(
                f"scorer criterion chunk size is frozen to {SCORER_CRITERION_CHUNK_SIZE}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["epsilon_grid"] = list(self.epsilon_grid)
        return result
