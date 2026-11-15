"""Mutation settings, separate from the network architecture."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    mutation_rate: float
    mutation_scale: float

    def __post_init__(self):
        if not math.isfinite(self.mutation_rate) or not 0 < self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in (0, 1]")
        if not math.isfinite(self.mutation_scale) or self.mutation_scale <= 0:
            raise ValueError("mutation_scale must be finite and positive")

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("ALLMENGINE_TRAINING_CONFIG")
        path = Path(path) if path else Path(__file__).with_name("training.json")
        return cls(**json.loads(path.read_text(encoding="utf-8")))
