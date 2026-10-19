from __future__ import annotations

from pydantic import BaseModel, Field


class SwarmAssessment(BaseModel):
    task_type: str
    complexity: float = Field(ge=0.0, le=1.0)
    parallelizability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    stakes: float = Field(ge=0.0, le=1.0)
    swarm_required: bool


def choose_worker_target(
    assessment: SwarmAssessment,
    *,
    minimum: int = 10,
    maximum: int = 70,
) -> int:
    """Choose a bounded worker budget while avoiding overscaling sequential work."""
    if not assessment.swarm_required:
        return 1

    score = min(
        1.0,
        assessment.complexity * (0.55 + 0.45 * assessment.parallelizability)
        + 0.15 * assessment.uncertainty * assessment.parallelizability
        + 0.10 * assessment.stakes * assessment.parallelizability,
    )
    # Tier fractions are calibrated on a 70-worker ceiling and scale to any maximum.
    tiers = ((0.20, 0.0), (0.40, 20 / 70), (0.60, 30 / 70), (0.75, 45 / 70), (0.90, 60 / 70))
    fraction = 1.0
    for threshold, tier_fraction in tiers:
        if score < threshold:
            fraction = tier_fraction
            break
    proposed = max(minimum, round(maximum * fraction))

    parallel_cap = minimum + round((maximum - minimum) * assessment.parallelizability)
    return max(minimum, min(maximum, parallel_cap, proposed))
