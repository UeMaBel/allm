"""Supervised Adam optimization over generated or observed ALU operations."""
from collections import Counter
from copy import deepcopy

import torch
from torch import nn

from .fitness import operation_key
from .result import TrainingResult
from .targets import prepare_target, UnsupportedOperation


class GradientOptimizer:
    """Own Adam state across evaluations; targets never enter application execution."""

    def __init__(self, model, seed=0):
        self.model = model
        self.optimizer = torch.optim.Adam(model.module.parameters(), lr=model.training_config.learning_rate)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, operations):
        examples, excluded = [], Counter()
        distinct = {operation_key(op): op for op in operations}
        for operation in distinct.values():
            try:
                inputs, targets, _ = prepare_target(self.model, operation)
                examples.append((inputs, targets))
            except UnsupportedOperation as error:
                excluded[str(error)] += 1
        for reason, count in excluded.items():
            print(f"  Excluded training operations ({reason}): {count}", flush=True)
        if not examples:
            raise ValueError("No supported neural operations available for gradient training")
        inputs = torch.tensor([item[0] for item in examples], dtype=torch.float32)
        targets = torch.tensor([item[1] for item in examples], dtype=torch.float32)
        config = self.model.training_config
        for _ in range(config.epochs):
            order = (torch.randperm(len(inputs), generator=self.generator) if config.shuffle
                     else torch.arange(len(inputs)))
            total = 0.0
            for indices in order.split(config.batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss(self.model.forward_logits(inputs[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; reduce learning_rate")
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(indices)
        return total / len(inputs)


def train_gradients(model, evaluate, score, report, rng, max_evaluations):
    """Optimize continuously, retaining the best evaluated state on a shared corpus.

    Each evaluation after the initial one follows ``epochs`` supervised passes.
    The held-out validation set never enters this loop. Optimizer state is local
    to this run; checkpoints contain the retained model, not a resumable optimizer.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least one")
    print(f"Evaluation 1/{max_evaluations} (initial model)", flush=True)
    best = evaluate(model)
    report(best)
    corpus = {operation_key(op): op for op in best.operations}
    best_state = model.state_dict()
    incumbent = deepcopy(model)
    optimizer = GradientOptimizer(model, rng.randrange(2**63))
    evaluations = 1
    while not best.success and evaluations < max_evaluations:
        loss = optimizer.fit(corpus.values())
        print(f"Evaluation {evaluations + 1}/{max_evaluations}; training BCE: {loss:.6f}", flush=True)
        candidate = evaluate(model)
        previous_count = len(corpus)
        corpus.update((operation_key(op), op) for op in candidate.operations)
        if len(corpus) > previous_count:
            print(f"  Shared operation corpus expanded: {previous_count} -> {len(corpus)} calls", flush=True)
        candidate = score(model, candidate, corpus.values())
        incumbent.load_state_dict(best_state)
        best = score(incumbent, best, corpus.values())
        report(candidate)
        evaluations += 1
        if candidate.success or (candidate.accuracy, candidate.score) >= (best.accuracy, best.score):
            best, best_state = candidate, model.state_dict()
    model.load_state_dict(best_state)
    print(f"Training finished: {'converged' if best.success else 'evaluation budget exhausted'}; "
          f"{evaluations} evaluations; retained operation fitness: {best.score:.4%}", flush=True)
    return TrainingResult(best.success, evaluations, best, best_state)
