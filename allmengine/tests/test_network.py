from dataclasses import replace
import json
import random

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
    assert bool(NeuralValue(0)) is False
    # Nine ones win even with a zero final bit, including actual control flow.
    model.layers[-1]["biases"][:] = [1] * 9 + [-1] * 7
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


@pytest.mark.parametrize("legacy_version", [None, 1, 2, 99])
def test_checkpoint_always_uses_majority_regardless_of_legacy_metadata(legacy_version):
    model = BinaryNetwork(small_config())
    for row in model.layers[-1]["weights"]:
        row[:] = [0] * len(row)
    model.layers[-1]["biases"][:] = [1, 1, 1, -1]
    saved = model.state_dict()
    assert "version" not in saved
    if legacy_version is not None:
        saved["version"] = legacy_version
    restored = BinaryNetwork()
    restored.load_state_dict(saved)
    assert restored.forward(OpCode.BOOL, 0) is True
    assert "version" not in restored.state_dict()


def test_actual_network_checkpoint_reaches_pytest_worker(tmp_path):
    model = BinaryNetwork(small_config())
    for row in model.layers[-1]["weights"]:
        row[:] = [0] * len(row)
    model.layers[-1]["biases"][:] = [1, 1, 1, -1]
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
    {"mutation_rate": 0}, {"mutation_rate": 1.1},
    {"mutation_scale": 0}, {"mutation_scale": float("nan")},
])
def test_reject_invalid_training_config(changes):
    with pytest.raises(ValueError):
        replace(TrainingConfig.load(), **changes)


def test_training_file_controls_mutations(tmp_path, monkeypatch):
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"mutation_rate": 1, "mutation_scale": 0.5}))
    monkeypatch.setenv("ALLMENGINE_TRAINING_CONFIG", str(path))
    model = BinaryNetwork(small_config())
    before = model.state_dict()
    rng = random.Random(42)
    expected = before["layers"][0]["weights"][0][0]
    rng.random()
    expected += rng.gauss(0, 0.5)
    model.mutate(random.Random(42))
    assert model.layers[0]["weights"][0][0] == expected
    restored = BinaryNetwork(small_config(), TrainingConfig(0.1, 0.01))
    restored.load_state_dict(model.state_dict())
    assert restored.training_config == TrainingConfig(1, 0.5)
    assert BinaryNetwork(small_config(), path).training_config == TrainingConfig(1, 0.5)


def test_load_legacy_mutation_settings_without_changing_checkpoint():
    model = BinaryNetwork(small_config())
    state = model.state_dict()
    del state["training_config"]
    state["config"].update(mutation_rate=0.2, mutation_scale=0.3)
    model.load_state_dict(state)
    assert model.training_config == TrainingConfig(0.2, 0.3)
    assert state["config"]["mutation_rate"] == 0.2
    assert "mutation_rate" not in model.state_dict()["config"]
