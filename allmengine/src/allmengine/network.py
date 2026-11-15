"""PyTorch ReLU network with a binary input/output runtime interface."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import torch
from torch import nn

from .constants import OPCODE_IDS, PREDICATES, OpCode
from .training_config import TrainingConfig


@dataclass(frozen=True)
class NetworkConfig:
    input_bits: int
    hidden_bits: tuple[int, ...]
    output_bits: int
    opcode_bits: int
    operand_bits: int
    seed: int

    def __post_init__(self):
        object.__setattr__(self, "hidden_bits", tuple(self.hidden_bits))
        sizes = (self.input_bits, *self.hidden_bits, self.output_bits,
                 self.opcode_bits, self.operand_bits)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("All bit widths must be positive integers")
        if self.input_bits != self.opcode_bits + 2 * self.operand_bits:
            raise ValueError("input_bits must equal opcode_bits + 2 * operand_bits")
        if (1 << self.opcode_bits) < len(OPCODE_IDS):
            raise ValueError("opcode_bits cannot represent every opcode")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @property
    def widths(self):
        return (self.input_bits, *self.hidden_bits, self.output_bits)

    @classmethod
    def load(cls, path=None):
        path = path or os.environ.get("ALLMENGINE_NETWORK_CONFIG")
        path = Path(path) if path else Path(__file__).with_name("network.json")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _bits(value, width):
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


class BinaryNetwork:
    """Shared CPU network: ReLU hidden units, linear logits, hard output bits.

    The historical name describes the external interface, not hidden activations.
    ``module`` is the differentiable PyTorch implementation; JSON checkpoints
    remain separate from PyTorch's tensor state dictionaries.
    """

    MODEL_TYPE = "relu-bit-network-v1"

    def __init__(self, config=None, training_config=None):
        self.config = config if isinstance(config, NetworkConfig) else NetworkConfig.load(config)
        self.training_config = (training_config if isinstance(training_config, TrainingConfig)
                                else TrainingConfig.load(training_config))
        self.module = self._build(self.config)

    @staticmethod
    def _build(config):
        # Local initialization must not change the application's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            modules = []
            for index, (inputs, outputs) in enumerate(zip(config.widths, config.widths[1:])):
                modules.append(nn.Linear(inputs, outputs, device="cpu", dtype=torch.float32))
                if index < len(config.hidden_bits):
                    modules.append(nn.ReLU())
            return nn.Sequential(*modules)

    def forward_logits(self, inputs):
        """Differentiable scores for float tensors shaped (..., input_bits)."""
        return self.module(inputs)

    def forward_bits(self, bits):
        values = list(bits)
        if len(values) != self.config.input_bits:
            raise ValueError(f"Expected {self.config.input_bits} input bits")
        if any(type(bit) not in (int, bool) or bit not in (0, 1) for bit in values):
            raise ValueError("Inputs must be binary values")
        with torch.inference_mode():
            logits = self.forward_logits(torch.tensor(values, dtype=torch.float32))
            return (logits >= 0).to(torch.int64).tolist()

    def encode(self, opcode, lhs, rhs=None):
        """MSB-first opcode + two signed two's-complement integer operands."""
        width = self.config.operand_bits
        operands = []
        for value in (lhs, 0 if rhs is None else rhs):
            if type(value) not in (int, bool):
                raise TypeError("The initial network codec supports only integers and bools")
            if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
                raise OverflowError(f"Operand does not fit in signed {width} bits")
            operands.extend(_bits(value, width))
        return _bits(OPCODE_IDS[OpCode(opcode)], self.config.opcode_bits) + operands

    def forward(self, opcode, lhs, rhs=None):
        opcode = OpCode(opcode)
        bits = self.forward_bits(self.encode(opcode, lhs, rhs))
        return self.decode(opcode, bits)

    def decode(self, opcode, bits):
        """Decode raw output bits using the same rules for runtime and training."""
        if OpCode(opcode) in PREDICATES:
            # Strict majority at any configured output width; ties are false.
            return sum(bits) * 2 > len(bits)
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value - (1 << len(bits)) if bits[0] else value

    def state_dict(self):
        return {"model_type": self.MODEL_TYPE, "config": asdict(self.config),
                "training_config": asdict(self.training_config),
                "layers": [{"weights": layer.weight.detach().tolist(),
                            "biases": layer.bias.detach().tolist()}
                           for layer in self.module if isinstance(layer, nn.Linear)]}

    def load_state_dict(self, state):
        if state.get("model_type") != self.MODEL_TYPE:
            raise ValueError("Unsupported checkpoint model_type; legacy binary-activation "
                             "checkpoints require retraining with the ReLU network")
        config = NetworkConfig(**state["config"])
        training_config = TrainingConfig(**state["training_config"])
        layers = deepcopy(state["layers"])
        if len(layers) != len(config.widths) - 1:
            raise ValueError("Checkpoint layer count does not match config")
        for layer, inputs, outputs in zip(layers, config.widths, config.widths[1:]):
            weights, biases = layer["weights"], layer["biases"]
            if len(weights) != outputs or len(biases) != outputs:
                raise ValueError("Checkpoint output width does not match config")
            if any(len(row) != inputs for row in weights):
                raise ValueError("Checkpoint input width does not match config")
            for number in [*biases, *(value for row in weights for value in row)]:
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise ValueError("Checkpoint weights and biases must be finite numbers")
        module = self._build(config)
        with torch.no_grad():
            for layer, saved in zip((m for m in module if isinstance(m, nn.Linear)), layers):
                weight = torch.tensor(saved["weights"], dtype=torch.float32)
                bias = torch.tensor(saved["biases"], dtype=torch.float32)
                if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                    raise ValueError("Checkpoint parameters exceed float32 range")
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
        self.config, self.module, self.training_config = config, module, training_config
