from dataclasses import replace
import json
import sys

import pytest
import torch

from allmengine import BinaryNetwork, NetworkConfig, TrainingConfig
from allmengine.training.config import configure_training
from allmengine.training.fitness import evaluate_operations
from allmengine.training.gradient import GradientOptimizer
from allmengine.training.pretrain import main
from allmengine.training.targets import prepare_target


@pytest.fixture
def model():
    return BinaryNetwork(
        replace(NetworkConfig.load(), input_bits=12, operand_bits=2,
                hidden_bits=(32,), output_bits=4),
        TrainingConfig(learning_rate=0.03, batch_size=8, epochs=200),
    )


def test_adam_learns_small_signed_addition_table_reproducibly(model):
    operations = [dict(opcode="add", lhs=a, rhs=b) for a in range(-2, 2) for b in range(-2, 2)]
    before = model.state_dict()
    assert evaluate_operations(model, operations).accuracy < 1
    inputs = torch.tensor([model.encode("add", op["lhs"], op["rhs"]) for op in operations], dtype=torch.float32)
    targets = torch.tensor([prepare_target(model, op)[1] for op in operations], dtype=torch.float32)
    initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(model.forward_logits(inputs), targets).item()
    loss = GradientOptimizer(model, seed=7).fit(operations)
    assert loss < initial_loss / 10
    assert evaluate_operations(model, operations).accuracy == 1
    learned = model.state_dict()
    model.load_state_dict(before)
    GradientOptimizer(model, seed=7).fit(operations)
    assert model.state_dict() == learned


def test_targets_preserve_integer_and_majority_boolean_encodings(model):
    assert prepare_target(model, dict(opcode="sub", lhs=0, rhs=1))[1] == [1, 1, 1, 1]
    assert prepare_target(model, dict(opcode="eq", lhs=1, rhs=1))[1] == [1, 1, 1, 1]
    assert prepare_target(model, dict(opcode="eq", lhs=0, rhs=1))[1] == [0, 0, 0, 0]


def test_unsupported_corpus_fails_clearly_without_updating(model, capsys):
    before = model.state_dict()
    with pytest.raises(ValueError, match="No supported neural operations"):
        GradientOptimizer(model).fit([dict(opcode="truediv", lhs=1, rhs=0)])
    assert "reference exception" in capsys.readouterr().out
    assert model.state_dict() == before


def test_settings_precedence_checkpoint_then_explicit_file_then_cli(model, tmp_path):
    path = tmp_path / "training.json"
    path.write_text(json.dumps(dict(learning_rate=0.02, batch_size=4, epochs=5, shuffle=False)))
    configure_training(model, path, dict(epochs=2))
    assert model.training_config == TrainingConfig(learning_rate=0.02, batch_size=4, epochs=2, shuffle=False)
    configure_training(model, overrides=dict(learning_rate=0.01))
    assert model.training_config.learning_rate == 0.01
    assert model.training_config.epochs == 2


def test_pretrain_cli_writes_relu_checkpoint_with_overridden_settings(model, tmp_path, monkeypatch):
    config = tmp_path / "network.json"
    config.write_text(json.dumps(model.state_dict()["config"]))
    checkpoint = tmp_path / "trained.json"
    monkeypatch.setenv("ALLMENGINE_NETWORK_CONFIG", str(config))
    monkeypatch.setattr(sys, "argv", ["pretrain", "--examples", "6", "--validation-examples", "2",
                                    "--minimum", "-2", "--maximum", "1", "--operations", "add",
                                    "--epochs", "2", "--batch-size", "3", "--learning-rate", "0.02",
                                    "--no-shuffle", "--max-evaluations", "2", "--checkpoint", str(checkpoint)])
    assert main() in (0, 1)
    state = json.loads(checkpoint.read_text())
    assert state["model_type"] == BinaryNetwork.MODEL_TYPE
    assert state["training_config"] == dict(learning_rate=0.02, batch_size=3, epochs=2, shuffle=False)
    model.load_state_dict(state)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e100])
def test_invalid_checkpoint_parameters_do_not_change_model(model, value):
    before = model.state_dict()
    invalid = model.state_dict()
    invalid["layers"][0]["weights"][0][0] = value
    with pytest.raises(ValueError):
        model.load_state_dict(invalid)
    assert model.state_dict() == before
