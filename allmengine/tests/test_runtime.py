import operator
import pytest
from allmengine import NeuralValue, OpCode, UniversalNeuralALU, alu
from allmengine.cpu import UnimplementedNetwork


@pytest.fixture(autouse=True)
def restore_model():
    previous = alu.model
    yield
    alu.set_model(previous)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def forward(self, opcode, lhs, rhs=None):
        self.calls.append((opcode, lhs, rhs))
        return True if opcode is OpCode.BOOL else 123


@pytest.mark.parametrize("name", [
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "matmul",
    "lshift", "rshift", "and_", "or_", "xor", "lt", "le", "eq", "ne", "gt", "ge",
])
def test_binary_operations_use_model_output(name):
    model = RecordingModel()
    alu.set_model(model)
    result = getattr(operator, name)(NeuralValue(7), NeuralValue(2))
    assert result.value == 123
    assert model.calls == [(OpCode(name), 7, 2)]


@pytest.mark.parametrize("name", [
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
    "lshift", "rshift", "and_", "or_", "xor",
])
def test_reflected_operand_order(name):
    model = RecordingModel()
    alu.set_model(model)
    assert getattr(operator, name)(9, NeuralValue(2)).value == 123
    assert model.calls == [(OpCode(name), 9, 2)]


@pytest.mark.parametrize("name", ["neg", "pos", "abs", "invert"])
def test_unary_delegation(name):
    model = RecordingModel()
    alu.set_model(model)
    assert getattr(operator, name)(NeuralValue(7)).value == 123
    assert model.calls == [(OpCode(name), 7, None)]


def test_existing_values_share_singleton_and_control_flow():
    value = NeuralValue(7)
    model = RecordingModel()
    UniversalNeuralALU().set_model(model)
    assert UniversalNeuralALU() is alu
    assert bool(value < 2) is True
    assert model.calls == [(OpCode.LT, 7, 2), (OpCode.BOOL, 123, None)]


def test_default_stub_does_not_fall_back_to_native_math():
    alu.set_model(UnimplementedNetwork())
    with pytest.raises(NotImplementedError):
        NeuralValue(1) + 2


def test_truthiness_requires_decoded_bool():
    class BadModel:
        def forward(self, *args):
            return 1
    alu.set_model(BadModel())
    with pytest.raises(TypeError, match="Python bool"):
        bool(NeuralValue(1))
