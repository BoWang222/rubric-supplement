from __future__ import annotations
from typing import Sequence
from .schema import RubricCriterion

def rubricarrow_items(criteria: Sequence[RubricCriterion]) -> str:
    return "\n".join(
        f"{index}. {criterion.criterion}. {criterion.guidance} "
        f"Anchors: {' | '.join(criterion.anchors)}"
        for index, criterion in enumerate(criteria, start=1)
    )
