# allmengine

An experimental Python runtime that routes numeric operators in selected
application packages through one shared neural ALU. Activate it once; keep your
application's normal `+`, `*`, comparisons, and `if` statements.

## Install

Requires Python 3.10 or newer. In your application's virtual environment, install
from this package directory:

```sh
python -m pip install -e .
```

From elsewhere, replace `.` with the path to the `allmengine` directory containing
`pyproject.toml`. Installation alone does not change Python's operators.

## Use with an existing application

Activate instrumentation in a separate launcher before importing your code:

```python
import allmengine

allmengine.enable("myapp")

from myapp.main import main
main()
```

Your application needs no `NeuralValue` constructors:

```python
def calculate(left, right):
    total = left + right
    if total > 0:
        return total * 2
    return 0
```

Alternatively, run an importable module directly:

```sh
python -m allmengine myapp.main
```

The runner selects the entry module's top-level package. To select additional
packages, put options before the entry module; application arguments follow it:

```sh
python -m allmengine --package myapp --package sharedlogic myapp.main --app-option
```

For a standalone `script.py`, run `python -m allmengine script` from its directory.
Calling `enable()` inside an already-running module does not rewrite that module.

## Model and configuration

The default `BinaryNetwork` starts untrained. Its outputs are not guaranteed to
implement arithmetic correctly. It has 40 input bits, three 128-bit hidden
layers, and 16 output bits, using binary activations and floating-point weights.

Edit [network.json](src/allmengine/network.json), set
`ALLMENGINE_NETWORK_CONFIG` to an absolute config path before starting Python,
or install a model explicitly:

```python
from allmengine import BinaryNetwork, alu

alu.set_model(BinaryNetwork("path/to/network.json"))
```

The input layout is **8-bit opcode → 16-bit left operand → 16-bit right operand**.
The network codec accepts bools and signed integers; default operands must fit
in 16 bits. Strings, character types, floats, and exception signaling have no
network encoding. Configuration changes apply to newly created models.

Runtime operations with a built-in float operand use native Python, including
mixed int/float arithmetic, comparisons, unary operations, and float truthiness.
For example, `2 + 0.5` returns `2.5`. This also applies to `NeuralValue`. Native
errors still apply, including division by zero. Integer-only operations remain
neural: `3 / 2` uses the network, while `3 / 2.0` uses native division. A later
truth check on a comparison's boolean result still uses the bool ALU path.
Direct `BinaryNetwork.forward` and `encode` calls remain integer-only; runtime
fallback happens in `alu.execute` before the network is called.

Comparisons and truthiness decode by strict majority: with 16 output bits,
9–16 ones mean true and 0–8 mean false. The threshold adapts to the configured
output width; ties are false. Arithmetic still decodes as signed integers.
Checkpoints store configuration and weights without versioning. Loaded models
always use majority decoding, including those saved by older implementations.

For a deterministic wiring check, explicitly select the native reference model
before importing the application:

```python
from allmengine import alu
from allmengine.reference import ReferenceModel

alu.set_model(ReferenceModel())
```

The reference model is a test oracle, not a neural model.

## Train using pytest

```sh
python -m allmengine.trainer /path/to/project --factory allmengine.network:BinaryNetwork --instrument myapp --max-evaluations 100 --checkpoint weights.json
```

Each candidate runs in a fresh worker. Select application packages with
`--instrument`, keeping assertions and expected values native. Repeat the flag
for more packages. Do not replace the candidate model in test setup.

The trainer keeps mutations with non-decreasing passed-test fractions and stops
on success or budget exhaustion. Checkpoints store configuration and weights.
Mutation search does not guarantee convergence. Choose tests whose inputs and
results fit the codec's range and supported types.

Custom factories must be importable zero-argument constructors implementing
`forward(opcode, lhs, rhs)`, `state_dict()`, `load_state_dict(state)`, and
`mutate(random.Random)`. State must be JSON-compatible.

## Scope

The import hook transforms source in memory without editing application files.
It supports numeric operator syntax, comparisons, control-flow truth checks,
short-circuit expressions, and augmented assignments. Plain Python inputs and
results cross the application boundary.

Engine internals, the standard library, and dependencies outside the selected
package names stay native, avoiding recursive neural execution. Keep custom
model implementations outside instrumented packages too.

Activation must precede imports. Only ordinary source modules are supported;
native extensions and dynamically compiled code are not rewritten. Library and
built-in calls such as `sum`, `abs`, and `bool` retain their native behavior.
Non-numeric object operations, identity, membership, and pattern matching stay
native, although explicit case guards are instrumented. Mixed comparison chains
containing identity or membership checks also stay native.

The returned activation handle supports `close()` and context-manager use.
Closing it stops future instrumentation; already-imported functions remain
instrumented until the process ends.

See the [project README](../README.md) for the complete integration guide,
development commands, and current implementation boundaries. The
[bit encoding specification](../BIT_ENCODING.md) describes the implemented input
layout, integer encoding, and majority-vote boolean decoding.
