"""One definition of supported reference targets for training and scoring."""
from ..constants import OpCode, PREDICATES
from ..reference import ReferenceModel


class UnsupportedOperation(ValueError):
    """The codec cannot represent this supervised example."""


def prepare_target(model, operation):
    opcode = OpCode(operation["opcode"])
    lhs, rhs = operation["lhs"], operation["rhs"]
    try:
        expected = (operation["expected"] if "expected" in operation else
                    ReferenceModel().forward(opcode, lhs, rhs))
    except (ArithmeticError, TypeError, ValueError) as error:
        raise UnsupportedOperation("reference exception") from error
    try:
        inputs = model.encode(opcode, lhs, rhs)
    except (TypeError, OverflowError) as error:
        raise UnsupportedOperation("unsupported operands") from error
    width = model.config.output_bits
    if opcode in PREDICATES:
        targets = [int(bool(expected))] * width
    else:
        if type(expected) is float and expected.is_integer():
            expected = int(expected)
        if type(expected) not in (int, bool):
            raise UnsupportedOperation("non-integer result")
        if not -(1 << (width - 1)) <= expected < (1 << (width - 1)):
            raise UnsupportedOperation("result overflow")
        targets = [(expected >> shift) & 1 for shift in range(width - 1, -1, -1)]
    return inputs, targets, expected
