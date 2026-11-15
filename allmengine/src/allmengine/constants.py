"""Shared operation names and stable network opcode IDs."""
import ast
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


BINARY_AST_OPS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow", ast.MatMult: "matmul",
    ast.LShift: "lshift", ast.RShift: "rshift", ast.BitAnd: "and_",
    ast.BitOr: "or_", ast.BitXor: "xor",
}
UNARY_AST_OPS = {ast.UAdd: "pos", ast.USub: "neg", ast.Invert: "invert"}
COMPARISON_AST_OPS = {ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne",
            ast.Gt: "gt", ast.GtE: "ge"}
