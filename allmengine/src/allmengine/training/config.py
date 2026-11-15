"""Adam settings and shared CLI overrides, separate from architecture."""
from dataclasses import dataclass, replace
import argparse
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    shuffle: bool = True
    time_budget_seconds: float = 0
    checkpoint_interval_seconds: float = 60

    def __post_init__(self):
        if (type(self.learning_rate) not in (int, float) or
                not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")
        for name in ("batch_size", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.shuffle) is not bool:
            raise ValueError("shuffle must be a boolean")
        for name in ("time_budget_seconds", "checkpoint_interval_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.checkpoint_interval_seconds == 0:
            raise ValueError("checkpoint_interval_seconds must be positive")

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("ALLMENGINE_TRAINING_CONFIG")
        path = Path(path) if path else Path(__file__).parent.parent / "training.json"
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def add_training_arguments(parser):
    parser.add_argument("--training-config", type=Path, help="Adam settings JSON; overrides checkpoint settings")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int, help="Epochs between evaluations (default: training.json)")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--time-budget", dest="time_budget_seconds", type=float,
                        help="Training time budget in seconds; 0 disables the limit")
    parser.add_argument("--checkpoint-interval", dest="checkpoint_interval_seconds", type=float,
                        help="Seconds between periodic saves to --checkpoint (default: 60)")


def training_overrides(args):
    return {name: getattr(args, name) for name in
            ("learning_rate", "batch_size", "epochs", "shuffle",
             "time_budget_seconds", "checkpoint_interval_seconds")
            if getattr(args, name) is not None}


def configure_training(model, path=None, overrides=None):
    if not hasattr(model, "training_config"):
        if overrides and set(overrides) - {"time_budget_seconds", "checkpoint_interval_seconds"}:
            raise ValueError("Optimizer settings require a gradient-trainable model")
        return replace(TrainingConfig.load(path), **(overrides or {}))
    base = TrainingConfig.load(path) if path is not None else model.training_config
    model.training_config = replace(base, **(overrides or {}))
    return model.training_config
