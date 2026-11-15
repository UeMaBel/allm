"""Reference-based operation feedback, independent of pytest assertions.

Tests supply the operations to observe. Native results are targets only: they
never replace a candidate's answers in the application.
"""
from collections import Counter
from dataclasses import dataclass
import json

from .constants import OpCode, PREDICATES
from .reference import ReferenceModel


BITWISE = {OpCode.AND, OpCode.OR, OpCode.XOR, OpCode.INVERT,
           OpCode.LSHIFT, OpCode.RSHIFT}


class RecordingModel:
    """Wrap a worker's model and capture the actual bits used by forward()."""

    def __init__(self, model):
        self.model = model
        self.operations = []
        self.bits = None
        if callable(getattr(model, "forward_bits", None)):
            forward_bits = model.forward_bits

            def record_bits(inputs):
                bits = forward_bits(inputs)
                self.bits = list(bits)
                return bits

            model.forward_bits = record_bits

    def __getattr__(self, name):
        # Test setup can inspect config or checkpoint methods as before.
        return getattr(self.model, name)

    def forward(self, opcode, lhs, rhs=None):
        self.bits = None
        entry = {"opcode": OpCode(opcode).value, "lhs": lhs, "rhs": rhs}
        try:
            return self.model.forward(opcode, lhs, rhs)
        finally:
            # Even a codec error identifies an operation the suite requires.
            entry["bits"] = self.bits
            self.operations.append(entry)


def operation_key(operation):
    """Deduplicate calls, keeping bool and int operands distinct."""
    return json.dumps([operation[name] for name in ("opcode", "lhs", "rhs")])


@dataclass(frozen=True)
class OperationFitness:
    score: float
    scored: int
    excluded: dict


def evaluate_operations(model, operations):
    """Replay a common corpus so branch changes cannot make fitness easier.

    Average normalized scores per distinct operation input, not per execution:
    repeating an easy operation in a loop must not dominate training.
    """
    reference = ReferenceModel()
    scores = []
    excluded = Counter()
    binary = all(callable(getattr(model, name, None))
                 for name in ("encode", "forward_bits", "decode"))
    for operation in operations:
        opcode = OpCode(operation["opcode"])
        lhs, rhs = operation["lhs"], operation["rhs"]
        try:
            expected = reference.forward(opcode, lhs, rhs)
        except (ArithmeticError, TypeError, ValueError):
            excluded["reference exception"] += 1
            continue
        if binary:
            try:
                inputs = model.encode(opcode, lhs, rhs)
            except (TypeError, OverflowError):
                excluded["unsupported operands"] += 1
                continue
            width = model.config.output_bits
            if opcode not in PREDICATES:
                # True division can still have an integral, representable target.
                if type(expected) is float and expected.is_integer():
                    expected = int(expected)
                if type(expected) not in (int, bool):
                    excluded["non-integer result"] += 1
                    continue
                if not -(1 << (width - 1)) <= expected < (1 << (width - 1)):
                    excluded["result overflow"] += 1
                    continue
            bits = model.forward_bits(inputs)
            if opcode in PREDICATES:
                # Respect majority decoding: any correct majority earns full credit.
                score = float(model.decode(opcode, bits) == expected)
            else:
                targets = [(expected >> shift) & 1 for shift in range(width - 1, -1, -1)]
                weights = ([1] * width if opcode in BITWISE else
                           [1 << shift for shift in range(width - 1, -1, -1)])
                score = sum(w for bit, target, w in zip(bits, targets, weights)
                            if bit == target) / sum(weights)
        else:
            # Custom models without a bit interface still get operation-level
            # reference feedback, never the overall pytest pass fraction.
            try:
                score = float(model.forward(opcode, lhs, rhs) == expected)
            except (ArithmeticError, TypeError, ValueError):
                score = 0.0
        scores.append(score)
    return OperationFitness(sum(scores) / len(scores) if scores else 0.0,
                            len(scores), dict(excluded))
