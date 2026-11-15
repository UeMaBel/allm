"""Result shared by gradient training and custom-model mutation search."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingResult:
    converged: bool
    evaluations: int
    feedback: object
    state: dict
    stop_reason: str = "evaluation_budget"
