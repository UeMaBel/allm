"""Compatibility CLI for pytest fine-tuning."""
from .training.pytest_runner import Feedback, Trainer, TrainableModel, _create_model, main
from .training.result import TrainingResult

if __name__ == "__main__":
    raise SystemExit(main())
