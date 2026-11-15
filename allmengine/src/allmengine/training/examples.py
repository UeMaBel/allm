"""Reproducible integer examples with precomputed reference targets."""
from dataclasses import asdict, dataclass
import random
from ..constants import OpCode
from ..reference import ReferenceModel


@dataclass(frozen=True)
class OperationExample:
    opcode: str
    lhs: int
    rhs: int
    expected: int

    def as_operation(self):
        return asdict(self)


def generate_examples(config, count, operations=("add", "sub", "mul"),
                      minimum=-16, maximum=16, seed=0, exclude=()):
    """Generate distinct representable calls; fail on an exhausted domain."""
    if count < 1 or minimum > maximum or not operations:
        raise ValueError("Provide a positive count, nonempty operations and valid operand range")
    limit = 1 << (config.operand_bits - 1)
    if minimum < -limit or maximum >= limit:
        raise ValueError("Operand range exceeds network codec")
    opcodes = tuple(OpCode(op) for op in operations)
    if any(op in (OpCode.POW, OpCode.MATMUL) for op in opcodes):
        raise ValueError("Generated power and matrix multiplication are not supported yet")
    rng = random.Random(seed)
    oracle = ReferenceModel()
    seen = {(e.opcode, e.lhs, e.rhs) for e in exclude}
    result = []
    for _ in range(max(1000, count * 100)):
        op = rng.choice(opcodes)
        lhs, rhs = rng.randint(minimum, maximum), rng.randint(minimum, maximum)
        if op in (OpCode.NEG, OpCode.POS, OpCode.ABS, OpCode.INVERT, OpCode.BOOL):
            rhs = 0
        if op in (OpCode.LSHIFT, OpCode.RSHIFT) and not 0 <= rhs < config.output_bits:
            continue
        key = (op.value, lhs, rhs)
        if key in seen:
            continue
        try:
            expected = oracle.forward(op, lhs, rhs)
        except (ArithmeticError, TypeError, ValueError):
            continue
        if type(expected) is float and expected.is_integer():
            expected = int(expected)
        if type(expected) not in (int, bool):
            continue
        if not -(1 << (config.output_bits - 1)) <= expected < (1 << (config.output_bits - 1)):
            continue
        seen.add(key)
        result.append(OperationExample(*key, expected))
        if len(result) == count:
            return tuple(result)
    raise ValueError("Cannot generate enough distinct supported examples; expand the range or reduce count")
