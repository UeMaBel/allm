from dataclasses import replace

import pytest

from allmengine import BinaryNetwork, NetworkConfig, OpCode
from allmengine.fitness import RecordingModel, evaluate_operations


def constant_network(value, width=8):
    config = replace(NetworkConfig.load(), hidden_bits=(2,), output_bits=width)
    model = BinaryNetwork(config)
    state = model.state_dict()
    for row in state["layers"][-1]["weights"]:
        row[:] = [0] * len(row)
    state["layers"][-1]["biases"][:] = [
        1 if (value >> shift) & 1 else -1 for shift in range(width - 1, -1, -1)
    ]
    model.load_state_dict(state)
    return model


def call(opcode, lhs=0, rhs=0):
    return {"opcode": opcode.value, "lhs": lhs, "rhs": rhs}


def test_arithmetic_weights_bits_by_value():
    operations = [call(OpCode.ADD)]
    wrong_32 = evaluate_operations(constant_network(32), operations)
    wrong_4 = evaluate_operations(constant_network(4), operations)
    assert wrong_32.score == pytest.approx(1 - 32 / 255)
    assert wrong_4.score == pytest.approx(1 - 4 / 255)
    assert wrong_4.score > wrong_32.score


@pytest.mark.parametrize("opcode", [OpCode.AND, OpCode.OR, OpCode.XOR,
                                    OpCode.LSHIFT, OpCode.RSHIFT])
def test_bitwise_errors_have_equal_weights(opcode):
    operations = [call(opcode)]
    assert evaluate_operations(constant_network(32), operations).score == 7 / 8
    assert evaluate_operations(constant_network(4), operations).score == 7 / 8


def test_negative_targets_and_sign_bit():
    operations = [call(OpCode.SUB, 0, 1)]
    assert evaluate_operations(constant_network(-1), operations).score == 1
    assert evaluate_operations(constant_network(127), operations).score == pytest.approx(127 / 255)


def test_predicates_respect_majority_not_integer_encoding():
    operations = [call(OpCode.EQ, 1, 1)]
    assert evaluate_operations(constant_network(31), operations).score == 1
    assert evaluate_operations(constant_network(15), operations).score == 0


def test_unsupported_operations_are_explicitly_excluded():
    operations = [call(OpCode.ADD, 32768, 0), call(OpCode.ADD, 127, 1),
                  call(OpCode.TRUEDIV, 1, 2), call(OpCode.TRUEDIV, 1, 0)]
    result = evaluate_operations(constant_network(0), operations)
    assert result.scored == 0
    assert result.excluded == {"unsupported operands": 1, "result overflow": 1,
                               "non-integer result": 1, "reference exception": 1}


def test_recorder_preserves_actual_answers_and_records_raw_bits_and_errors():
    recorder = RecordingModel(constant_network(32))
    assert recorder.forward(OpCode.ADD, 3, 4) == 32
    assert recorder.operations[0] == {"opcode": "add", "lhs": 3, "rhs": 4,
                                      "bits": [0, 0, 1, 0, 0, 0, 0, 0]}
    with pytest.raises(OverflowError):
        recorder.forward(OpCode.ADD, 32768, 0)
    assert recorder.operations[-1]["bits"] is None
