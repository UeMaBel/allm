"""Explicit native oracle for wiring tests, NOT a neural implementation."""
import operator
from .cpu import OpCode


class ReferenceModel:
    def forward(self, opcode, lhs, rhs=None):
        if opcode is OpCode.BOOL:
            return bool(lhs)
        operation = getattr(operator, opcode.value)
        if opcode in (OpCode.NEG, OpCode.POS, OpCode.ABS, OpCode.INVERT):
            return operation(lhs)
        return operation(lhs, rhs)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state != {}:
            raise ValueError("ReferenceModel has no weights")
