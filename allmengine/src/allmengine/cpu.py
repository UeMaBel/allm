"""Single execution boundary for the shared neural network."""
import operator
from typing import Any, Protocol

from .constants import OpCode


class NetworkModel(Protocol):
    def forward(self, opcode: OpCode, lhs: Any, rhs: Any = None) -> Any:
        """Encode inputs, run the shared network, and decode a raw result."""
        ...


class UnimplementedNetwork:
    def forward(self, opcode, lhs, rhs=None):
        raise NotImplementedError("Install a network model on the global ALU first")


class UniversalNeuralALU:
    """One model per process, shared by already-created NeuralValues too."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            from .network import BinaryNetwork
            model = BinaryNetwork()
            cls._instance = super().__new__(cls)
            cls._instance.model = model
        return cls._instance

    def set_model(self, model: NetworkModel) -> None:
        self.model = model

    def execute(self, opcode: OpCode, lhs: Any, rhs: Any = None) -> Any:
        opcode = OpCode(opcode)
        # Floats bypass the integer-only network, including mixed int/float
        # operands. Keep this policy shared by instrumentation and NeuralValue.
        if type(lhs) is float or type(rhs) is float:
            if opcode is OpCode.BOOL:
                return bool(lhs)
            operation = getattr(operator, opcode.value)
            if opcode in (OpCode.NEG, OpCode.POS, OpCode.ABS, OpCode.INVERT):
                return operation(lhs)
            return operation(lhs, rhs)
        return self.model.forward(opcode, lhs, rhs)


alu = UniversalNeuralALU()
