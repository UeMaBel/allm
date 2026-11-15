"""Reference-based operation feedback, independent of pytest assertions.

Tests supply the operations to observe. Native results are targets only: they
never replace a candidate's answers in the application.
"""
from collections import Counter
from dataclasses import dataclass
import json

from ..constants import OpCode, PREDICATES
from ..reference import ReferenceModel
from .targets import prepare_target, UnsupportedOperation


BITWISE = {OpCode.AND, OpCode.OR, OpCode.XOR, OpCode.INVERT,
           OpCode.LSHIFT, OpCode.RSHIFT}


def operation_key(operation):
    """Deduplicate calls, keeping bool and int operands distinct."""
    return json.dumps([operation[name] for name in ("opcode", "lhs", "rhs")])


@dataclass(frozen=True)
class OperationFitness:
    score: float
    scored: int
    excluded: dict
    accuracy: float = 0.0


def evaluate_operations(model, operations):
    """Replay a common corpus so branch changes cannot make fitness easier.

    Average normalized scores per distinct operation input, not per execution:
    repeating an easy operation in a loop must not dominate training.
    """
    reference = ReferenceModel()
    scores = []
    correct = 0
    excluded = Counter()
    binary = all(callable(getattr(model, name, None))
                 for name in ("encode", "forward_bits", "decode"))
    for operation in operations:
        opcode = OpCode(operation["opcode"])
        lhs, rhs = operation["lhs"], operation["rhs"]
        if binary:
            try:
                inputs, targets, expected = prepare_target(model, operation)
            except UnsupportedOperation as error:
                excluded[str(error)] += 1
                continue
            width = model.config.output_bits
            bits = model.forward_bits(inputs)
            correct += model.decode(opcode, bits) == expected
            if opcode in PREDICATES:
                # Respect majority decoding: any correct majority earns full credit.
                score = float(model.decode(opcode, bits) == expected)
            else:
                weights = ([1] * width if opcode in BITWISE else
                           [1 << shift for shift in range(width - 1, -1, -1)])
                score = sum(w for bit, target, w in zip(bits, targets, weights)
                            if bit == target) / sum(weights)
        else:
            # Custom models without a bit interface still get operation-level
            # reference feedback, never the overall pytest pass fraction.
            try:
                expected = (operation["expected"] if "expected" in operation else
                            reference.forward(opcode, lhs, rhs))
            except (ArithmeticError, TypeError, ValueError):
                excluded["reference exception"] += 1
                continue
            try:
                score = float(model.forward(opcode, lhs, rhs) == expected)
            except (ArithmeticError, TypeError, ValueError):
                score = 0.0
            correct += score
        scores.append(score)
    return OperationFitness(sum(scores) / len(scores) if scores else 0.0,
                            len(scores), dict(excluded), correct / len(scores) if scores else 0.0)
