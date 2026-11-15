"""Shared operation names and stable network opcode IDs."""
from enum import Enum


class OpCode(str, Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    TRUEDIV = "truediv"
    FLOORDIV = "floordiv"
    MOD = "mod"
    POW = "pow"
    MATMUL = "matmul"
    LSHIFT = "lshift"
    RSHIFT = "rshift"
    AND = "and_"
    OR = "or_"
    XOR = "xor"
    LT = "lt"
    LE = "le"
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    NEG = "neg"
    POS = "pos"
    ABS = "abs"
    INVERT = "invert"
    BOOL = "bool"


BINARY_NAMES = (
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "matmul",
    "lshift", "rshift", "and", "or", "xor",
)
COMPARISON_NAMES = ("lt", "le", "eq", "ne", "gt", "ge")
UNARY_NAMES = ("neg", "pos", "abs", "invert")

# Enum declaration order is the checkpoint encoding. Append new operations;
# never reorder existing ones.
OPCODE_IDS = {opcode: index for index, opcode in enumerate(OpCode)}
PREDICATES = frozenset(OpCode[name.upper()] for name in (*COMPARISON_NAMES, "bool"))
