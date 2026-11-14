from dataclasses import replace
import json
import random

import pytest

from allmengine import BinaryNetwork, NetworkConfig, NeuralValue, OpCode, alu
from allmengine.trainer import Trainer


def small_config():
    return replace(NetworkConfig.load(), input_bits=12, hidden_bits=(3, 2),
                   output_bits=4, opcode_bits=8, operand_bits=2)


def test_requested_architecture_and_binary_output():
    model = BinaryNetwork()
    assert model.config.widths == (40, 128, 128, 128, 16)
    assert [(len(layer["weights"][0]), len(layer["weights"]))
            for layer in model.layers] == [(40, 128), (128, 128), (128, 128), (128, 16)]
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


def test_known_weights_propagate_bits_through_every_layer():
    config = replace(small_config(), hidden_bits=(1, 1, 1), output_bits=1)
    model = BinaryNetwork(config)
    for layer in model.layers:
        for row in layer["weights"]:
            row[:] = [1] + [0] * (len(row) - 1)
        layer["biases"][:] = [-0.5]
    assert model.forward_bits([0] * 12) == [0]
    assert model.forward_bits([1] + [0] * 11) == [1]
    model.layers[1]["weights"][0][0] = -1
    assert model.forward_bits([1] + [0] * 11) == [0]


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
    for row in model.layers[-1]["weights"]:
        row[:] = [0] * len(row)
    model.layers[-1]["biases"][:] = [-1] * 13 + [1, -1, 1]
    monkeypatch.setattr(alu, "model", model)
    assert (NeuralValue(10) + 20).value == 5
    assert bool(NeuralValue(0)) is True


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
    {"mutation_rate": 0}, {"mutation_scale": float("nan")},
    {"input_bits": 36, "opcode_bits": 4},
])
def test_reject_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(NetworkConfig.load(), **changes)


def test_checkpoint_roundtrip_restores_config_weights_and_outputs():
    original = BinaryNetwork(small_config())
    saved = json.loads(json.dumps(original.state_dict()))
    original.mutate(random.Random(42))
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


def test_actual_network_checkpoint_reaches_pytest_worker(tmp_path):
    model = BinaryNetwork(small_config())
    expected = model.forward(OpCode.ADD, 1, 0)
    (tmp_path / "test_candidate.py").write_text(
        "from allmengine import NeuralValue, alu\n"
        "def test_candidate():\n"
        "    assert alu.model.config.widths == (12, 3, 2, 4)\n"
        f"    assert (NeuralValue(1) + 0).value == {expected}\n"
    )
    feedback = Trainer("allmengine.network:BinaryNetwork", tmp_path).evaluate(model)
    assert feedback.success
