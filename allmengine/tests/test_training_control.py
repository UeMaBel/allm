from dataclasses import replace
import json
import random

import pytest
import torch

from allmengine import BinaryNetwork, NetworkConfig, TrainingConfig
from allmengine.training.control import RunControl, save_checkpoint
from allmengine.training.fitness import evaluate_operations
from allmengine.training.gradient import train_gradients
from allmengine.training.pretrain import Feedback
from allmengine.training.pretrain import pretrain


class Clock:
    now = 0

    def __call__(self):
        return self.now


def test_periodic_save_uses_latest_retained_state_and_deadline(tmp_path):
    clock = Clock()
    path = tmp_path / "checkpoint.json"
    control = RunControl(TrainingConfig(time_budget_seconds=10, checkpoint_interval_seconds=3),
                         path, clock=clock)
    control.retain({"weight": 1})
    assert json.loads(path.read_text()) == {"weight": 1}
    clock.now = 2
    control.retain({"weight": 2})
    assert json.loads(path.read_text()) == {"weight": 1}
    clock.now = 3
    assert control.pulse()
    assert json.loads(path.read_text()) == {"weight": 2}
    clock.now = 10
    assert not control.pulse()
    assert control.reason(False) == "time_budget"


def test_failed_write_preserves_previous_checkpoint_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"weight": 1})
    with pytest.raises(ValueError):
        save_checkpoint(path, {"weight": float("nan")})
    assert json.loads(path.read_text()) == {"weight": 1}
    assert list(tmp_path.iterdir()) == [path]


def callbacks():
    operations = ({"opcode": "add", "lhs": 0, "rhs": 0},)

    def score(model, feedback, calls):
        fitness = evaluate_operations(model, calls)
        return replace(feedback, score=fitness.score, accuracy=fitness.accuracy)

    def evaluate(model):
        return score(model, Feedback(operations), operations)

    return evaluate, score


def incorrect_model():
    model = BinaryNetwork(replace(NetworkConfig.load(), hidden_bits=(), output_bits=4),
                          TrainingConfig(epochs=100, batch_size=1))
    state = model.state_dict()
    for row in state["layers"][0]["weights"]:
        row[:] = [0] * len(row)
    state["layers"][0]["biases"][:] = [1] * 4
    model.load_state_dict(state)
    return model


def test_budget_stops_between_minibatches_and_restores_evaluated_checkpoint(tmp_path, monkeypatch):
    model = incorrect_model()
    initial = model.state_dict()
    clock = Clock()
    path = tmp_path / "checkpoint.json"
    control = RunControl(TrainingConfig(time_budget_seconds=10), path, clock=clock)
    steps = []
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        steps.append(1)
        clock.now += 10
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    evaluate, score = callbacks()
    result = train_gradients(model, evaluate, score, lambda feedback: None, random.Random(0), 10, control)
    assert len(steps) == 1
    assert result.stop_reason == "time_budget"
    assert result.evaluations == 1
    assert not result.converged
    assert result.state == initial == model.state_dict()
    assert json.loads(path.read_text()) == json.loads(json.dumps(initial))


def test_interrupt_saves_and_restores_last_evaluated_state(tmp_path, monkeypatch):
    model = incorrect_model()
    initial = model.state_dict()
    path = tmp_path / "checkpoint.json"
    control = RunControl(model.training_config, path)
    evaluate, score = callbacks()

    def interrupt(optimizer, *args, **kwargs):
        with torch.no_grad():
            model.module[0].bias.fill_(999)
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.Adam, "step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_gradients(model, evaluate, score, lambda feedback: None, random.Random(0), 10, control)
    assert model.state_dict() == initial
    assert json.loads(path.read_text()) == json.loads(json.dumps(initial))


@pytest.mark.parametrize("changes", [
    {"time_budget_seconds": -1}, {"time_budget_seconds": float("nan")},
    {"time_budget_seconds": True}, {"checkpoint_interval_seconds": 0},
    {"checkpoint_interval_seconds": float("inf")},
])
def test_invalid_run_settings(changes):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)


def test_pretraining_budget_saves_and_skips_validation(tmp_path, capsys):
    model = incorrect_model()
    model.training_config = replace(model.training_config, time_budget_seconds=1e-9)
    path = tmp_path / "checkpoint.json"
    result = pretrain(model, examples=4, validation_examples=2, operations=("add",),
                      minimum=-2, maximum=1, max_evaluations=10, checkpoint=path)
    assert result.stop_reason == "time_budget"
    assert result.evaluations == 1
    assert "Validation skipped: time budget exhausted" in capsys.readouterr().out
    assert json.loads(path.read_text()) == json.loads(json.dumps(result.state))
