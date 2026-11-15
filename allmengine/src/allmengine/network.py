"""Dense network with binary activations and floating-point trainable weights."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random

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
    """One shared fully connected network; threshold each neuron at zero.

    Bit widths describe logical activations, not Python storage or weight sizes.
    Initial weights are random; this model does not know arithmetic in advance.
    """

    def __init__(self, config=None, training_config=None):
        self.config = config if isinstance(config, NetworkConfig) else NetworkConfig.load(config)
        self.training_config = (training_config if isinstance(training_config, TrainingConfig)
                                else TrainingConfig.load(training_config))
        rng = random.Random(self.config.seed)
        self.layers = []
        for inputs, outputs in zip(self.config.widths, self.config.widths[1:]):
            scale = 1 / math.sqrt(inputs)
            self.layers.append({
                "weights": [[rng.gauss(0, scale) for _ in range(inputs)]
                            for _ in range(outputs)],
                "biases": [rng.gauss(0, scale) for _ in range(outputs)],
            })

    def forward_bits(self, bits):
        values = list(bits)
        if len(values) != self.config.input_bits:
            raise ValueError(f"Expected {self.config.input_bits} input bits")
        if any(type(bit) not in (int, bool) or bit not in (0, 1) for bit in values):
            raise ValueError("Inputs must be binary values")
        for layer in self.layers:
            values = [int(sum(weight * bit for weight, bit in zip(row, values)) + bias >= 0)
                      for row, bias in zip(layer["weights"], layer["biases"])]
        return values

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
        if opcode in PREDICATES:
            # Strict majority at any configured output width; ties are false.
            return sum(bits) * 2 > len(bits)
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value - (1 << len(bits)) if bits[0] else value

    def state_dict(self):
        return {"config": asdict(self.config),
                "training_config": asdict(self.training_config),
                "layers": deepcopy(self.layers)}

    def load_state_dict(self, state):
        # Older checkpoints stored mutation settings alongside architecture.
        config_data = dict(state["config"])
        legacy_training = {key: config_data.pop(key) for key in
                           ("mutation_rate", "mutation_scale") if key in config_data}
        config = NetworkConfig(**config_data)
        training_data = state.get("training_config", legacy_training)
        training_config = (TrainingConfig(**training_data) if training_data
                           else self.training_config)
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
        self.config, self.layers = config, layers
        self.training_config = training_config

    def mutate(self, rng):
        parameters = [row for layer in self.layers
                      for row in [*layer["weights"], layer["biases"]]]
        changed = False
        for row in parameters:
            for index in range(len(row)):
                if rng.random() < self.training_config.mutation_rate:
                    row[index] += rng.gauss(0, self.training_config.mutation_scale)
                    changed = True
        if not changed:
            row = rng.choice(parameters)
            row[rng.randrange(len(row))] += rng.gauss(0, self.training_config.mutation_scale)
