"""Python operator adapters. No arithmetic is implemented here."""
from .constants import BINARY_NAMES, COMPARISON_NAMES, UNARY_NAMES, OpCode
from .cpu import alu


class NeuralValue:
    __hash__ = None

    def __init__(self, value):
        self.value = _unwrap(value)

    def __repr__(self):
        return f"NeuralValue({self.value!r})"

    def __bool__(self):
        result = alu.execute(OpCode.BOOL, self.value)
        if type(result) is not bool:
            raise TypeError("The ALU BOOL operation must return a Python bool")
        return result

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            raise TypeError("Three-argument pow is outside the binary ALU interface")
        return NeuralValue(alu.execute(OpCode.POW, self.value, _unwrap(other)))


def _unwrap(value):
    return value.value if isinstance(value, NeuralValue) else value


def _binary(opcode, reflected=False):
    def operation(self, other):
        lhs, rhs = self.value, _unwrap(other)
        if reflected:
            lhs, rhs = rhs, lhs
        return NeuralValue(alu.execute(opcode, lhs, rhs))
    return operation


def _unary(opcode):
    def operation(self):
        return NeuralValue(alu.execute(opcode, self.value))
    return operation


# Comparisons return wrappers: `if a < b` executes LT followed by BOOL.
for _name in BINARY_NAMES:
    if _name != "pow":
        setattr(NeuralValue, f"__{_name}__", _binary(OpCode[_name.upper()]))
    setattr(NeuralValue, f"__r{_name}__", _binary(OpCode[_name.upper()], True))

for _name in COMPARISON_NAMES:
    setattr(NeuralValue, f"__{_name}__", _binary(OpCode[_name.upper()]))

for _name in UNARY_NAMES:
    setattr(NeuralValue, f"__{_name}__", _unary(OpCode[_name.upper()]))
