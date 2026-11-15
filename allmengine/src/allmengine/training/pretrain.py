"""Train directly on generated examples without starting pytest workers."""
import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from ..network import BinaryNetwork
from .examples import generate_examples
from .fitness import evaluate_operations
from .gradient import train_gradients
from .config import add_training_arguments, configure_training, training_overrides


@dataclass(frozen=True)
class Feedback:
    operations: tuple
    score: float = 0.0
    accuracy: float = 0.0
    @property
    def success(self):
        return self.accuracy == 1.0


def pretrain(model, examples=100, validation_examples=50, operations=("add", "sub", "mul"),
             minimum=-16, maximum=16, seed=0, max_evaluations=100):
    training = generate_examples(model.config, examples, operations, minimum, maximum, seed)
    validation = generate_examples(model.config, validation_examples, operations,
                                   minimum, maximum, seed + 1, exclude=training)
    corpus = tuple(e.as_operation() for e in training)
    def score(candidate, feedback, calls):
        fitness = evaluate_operations(candidate, calls)
        return replace(feedback, score=fitness.score, accuracy=fitness.accuracy)
    def evaluate(candidate):
        return score(candidate, Feedback(corpus), corpus)
    def report(feedback):
        print(f"  Training fitness: {feedback.score:.4%}; exact accuracy: {feedback.accuracy:.4%}", flush=True)
    print(f"Pretraining: {len(training)} examples; {len(validation)} held-out examples", flush=True)
    result = train_gradients(model, evaluate, score, report, random.Random(seed), max_evaluations)
    held_out = evaluate_operations(model, [e.as_operation() for e in validation])
    print(f"Validation fitness: {held_out.score:.4%}; exact accuracy: {held_out.accuracy:.4%}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained.json"))
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--validation-examples", type=int, default=50)
    parser.add_argument("--operations", nargs="+", default=["add", "sub", "mul"])
    parser.add_argument("--minimum", type=int, default=-16)
    parser.add_argument("--maximum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-evaluations", type=int, default=100)
    args = parser.parse_args()
    model = BinaryNetwork()
    if args.weights:
        model.load_state_dict(json.loads(args.weights.read_text(encoding="utf-8")))
    configure_training(model, args.training_config, training_overrides(args))
    result = pretrain(model, args.examples, args.validation_examples, args.operations,
                      args.minimum, args.maximum, args.seed, args.max_evaluations)
    args.checkpoint.write_text(json.dumps(result.state, indent=2), encoding="utf-8")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
