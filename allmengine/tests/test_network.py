from dataclasses import replace
import json
import torch

import pytest

from allmengine import BinaryNetwork, NetworkConfig, NeuralValue, OpCode, alu
from allmengine import TrainingConfig
from allmengine.trainer import Trainer


def small_config():
    return replace(NetworkConfig.load(), input_bits=12, hidden_bits=(3, 2),
                   output_bits=4, opcode_bits=8, operand_bits=2)


def test_requested_architecture_and_binary_output():
    model = BinaryNetwork()
    assert model.config.widths == (40, 128, 128, 128, 16)
    assert [(len(layer["weights"][0]), len(layer["weights"]))
            for layer in model.state_dict()["layers"]] == [(40, 128), (128, 128), (128, 128), (128, 16)]
    bits = model.forward_bits([0, 1] * 20)
    assert len(bits) == 16
    assert set(bits) <= {0, 1}


def test_config_file_and_environment_change_architecture(tmp_path, monkeypatch):
    config_path = tmp_path / "network.json"
    state = BinaryNetwork(small_config()).state_dict()
    config_path.write_text(json.dumps(state["config"]))
    monkeypatch.setenv("ALLMENGINE_NETWORK_CONFIG", str(config_path))
    assert BinaryNetwork().config.widths == (12, 3, 2, 4)
    assert BinaryNetwork(config_path).config == small_config()


def test_relu_hidden_units_and_unbounded_signed_output_scores():
    config = replace(small_config(), hidden_bits=(1, 1, 1), output_bits=1)
    model = BinaryNetwork(config)
    saved = model.state_dict()
    for layer in saved["layers"]:
        for row in layer["weights"]:
            row[:] = [2] + [0] * (len(row) - 1)
        layer["biases"][:] = [-0.5]
    model.load_state_dict(saved)
    assert model.forward_bits([0] * 12) == [0]
    assert model.forward_bits([1] + [0] * 11) == [1]
    inputs = torch.tensor([[0] * 12, [1] + [0] * 11], dtype=torch.float32)
    assert model.forward_logits(inputs).tolist() == [[-0.5], [8.5]]
    model.forward_logits(inputs).sum().backward()
    assert model.module[0].weight.grad.abs().sum() > 0


def test_signed_operand_encoding_and_predicate_decoding(monkeypatch):
    model = BinaryNetwork()
    encoded = model.encode(OpCode.SUB, -1, 2)
    assert encoded[:8] == [0, 0, 0, 0, 0, 0, 0, 1]
    assert encoded[8:24] == [1] * 16
    assert encoded[24:] == [0] * 14 + [1, 0]
    monkeypatch.setattr(model, "forward_bits", lambda bits: [1] * 16)
    assert model.forward(OpCode.ADD, 1, 2) == -1
    assert model.forward(OpCode.BOOL, 0) is True
    monkeypatch.setattr(model, "forward_bits", lambda bits: [0] * 16)
    assert model.forward(OpCode.EQ, 1, 1) is False


def test_neural_values_use_actual_network(monkeypatch):
    model = BinaryNetwork()
    # Constant output 5, produced by final-layer biases, not native addition.
    state = model.state_dict()
    for row in state["layers"][-1]["weights"]:
        row[:] = [0] * len(row)
    state["layers"][-1]["biases"][:] = [-1] * 13 + [1, -1, 1]
    model.load_state_dict(state)
    monkeypatch.setattr(alu, "model", model)
    assert (NeuralValue(10) + 20).value == 5
    assert bool(NeuralValue(0)) is False
    # Nine ones win even with a zero final bit, including actual control flow.
    state["layers"][-1]["biases"][:] = [1] * 9 + [-1] * 7
    model.load_state_dict(state)
    branch = "false"
    if NeuralValue(0):
        branch = "true"
    assert branch == "true"


@pytest.mark.parametrize("opcode", [OpCode.LT, OpCode.LE, OpCode.EQ, OpCode.NE,
                                   OpCode.GT, OpCode.GE, OpCode.BOOL])
@pytest.mark.parametrize("ones,expected", [(0, False), (7, False), (8, False),
                                          (9, True), (16, True)])
def test_all_predicates_use_strict_majority_regardless_of_bit_order(monkeypatch, opcode, ones, expected):
    model = BinaryNetwork(replace(small_config(), output_bits=16))
    bits = [1] * ones + [0] * (16 - ones)
    monkeypatch.setattr(model, "forward_bits", lambda inputs: bits)
    assert model.forward(opcode, 0, 0) is expected
    bits.reverse()
    assert model.forward(opcode, 0, 0) is expected


@pytest.mark.parametrize("width,ones,expected", [
    (1, 0, False), (1, 1, True),
    (4, 2, False), (4, 3, True),
    (5, 2, False), (5, 3, True),
])
def test_majority_threshold_follows_configured_output_width(monkeypatch, width, ones, expected):
    model = BinaryNetwork(replace(small_config(), output_bits=width))
    monkeypatch.setattr(model, "forward_bits", lambda inputs: [1] * ones + [0] * (width - ones))
    assert model.forward(OpCode.BOOL, 0) is expected


@pytest.mark.parametrize("bits", [[0] * 39, [2] * 40, [0.0] * 40])
def test_reject_invalid_input_bits(bits):
    with pytest.raises(ValueError):
        BinaryNetwork().forward_bits(bits)


def test_codec_rejects_unrepresentable_operands():
    model = BinaryNetwork()
    with pytest.raises(OverflowError):
        model.forward(OpCode.ADD, 32768, 0)
    with pytest.raises(OverflowError):
        model.forward(OpCode.ADD, -32769, 0)
    with pytest.raises(TypeError):
        model.forward(OpCode.ADD, 0.5, 0)


@pytest.mark.parametrize("changes", [
    {"input_bits": 39}, {"hidden_bits": (0,)}, {"output_bits": -1},
    {"input_bits": 36, "opcode_bits": 4},
])
def test_reject_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(NetworkConfig.load(), **changes)


def test_checkpoint_roundtrip_restores_config_weights_and_outputs():
    original = BinaryNetwork(small_config())
    saved = json.loads(json.dumps(original.state_dict()))
    with torch.no_grad():
        original.module[0].weight.add_(1)
    assert original.state_dict()["layers"] != saved["layers"]
    restored = BinaryNetwork()
    restored.load_state_dict(saved)
    original.load_state_dict(saved)
    assert restored.config == small_config()
    assert restored.forward_bits([1] * 12) == original.forward_bits([1] * 12)
    saved["layers"][0]["weights"][0][0] = 999
    assert restored.state_dict()["layers"] != saved["layers"]


def test_invalid_checkpoint_leaves_model_unchanged():
    model = BinaryNetwork(small_config())
    before = model.state_dict()
    broken = model.state_dict()
    broken["layers"][0]["weights"][0].pop()
    with pytest.raises(ValueError, match="input width"):
        model.load_state_dict(broken)
    assert model.state_dict() == before


@pytest.mark.parametrize("model_type", [None, "binary-v1", "future-network"])
def test_incompatible_checkpoint_is_rejected_atomically(model_type):
    model = BinaryNetwork(small_config())
    before = model.state_dict()
    saved = model.state_dict()
    saved.pop("model_type")
    if model_type is not None:
        saved["model_type"] = model_type
    with pytest.raises(ValueError, match="checkpoint model_type"):
        model.load_state_dict(saved)
    assert model.state_dict() == before


def test_actual_network_checkpoint_reaches_pytest_worker(tmp_path):
    model = BinaryNetwork(small_config())
    state = model.state_dict()
    for row in state["layers"][-1]["weights"]:
        row[:] = [0] * len(row)
    state["layers"][-1]["biases"][:] = [1, 1, 1, -1]
    model.load_state_dict(state)
    expected = model.forward(OpCode.ADD, 1, 0)
    (tmp_path / "test_candidate.py").write_text(
        "from allmengine import NeuralValue, alu\n"
        "def test_candidate():\n"
        "    assert alu.model.config.widths == (12, 3, 2, 4)\n"
        f"    assert (NeuralValue(1) + 0).value == {expected}\n"
        "    assert bool(NeuralValue(0)) is True\n"
    )
    feedback = Trainer("allmengine.network:BinaryNetwork", tmp_path).evaluate(model)
    assert feedback.success


@pytest.mark.parametrize("changes", [
    {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"learning_rate": float("inf")}, {"learning_rate": True},
    {"batch_size": 0}, {"batch_size": 1.5}, {"epochs": 0},
    {"epochs": True}, {"shuffle": 1},
])
def test_reject_invalid_training_config(changes):
    with pytest.raises(ValueError):
        replace(TrainingConfig.load(), **changes)


def test_training_settings_file_environment_and_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"learning_rate": 0.02, "batch_size": 8,
                                "epochs": 3, "shuffle": False}))
    monkeypatch.setenv("ALLMENGINE_TRAINING_CONFIG", str(path))
    expected = TrainingConfig(learning_rate=0.02, batch_size=8, epochs=3, shuffle=False)
    model = BinaryNetwork(small_config())
    assert model.training_config == expected
    restored = BinaryNetwork(small_config(), TrainingConfig())
    restored.load_state_dict(model.state_dict())
    assert restored.training_config == expected
    assert BinaryNetwork(small_config(), path).training_config == expected


def test_initialization_is_reproducible_without_changing_torch_random_state():
    before = torch.random.get_rng_state()
    first = BinaryNetwork(small_config())
    second = BinaryNetwork(small_config())
    assert first.state_dict() == second.state_dict()
    assert torch.equal(before, torch.random.get_rng_state())
