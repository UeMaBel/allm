# allm

Bored with coding?
Your codebase is super clean, well tested and you understand everything?
Make it interesting again, use allm and let the chaos reign.

allm is an experimental project that routes selected Python operations through
one shared neural ALU, then mutates its weights using feedback from your tests.
Application code keeps ordinary operators; one-time activation selects the
packages to instrument.

The repository contains [allmengine](allmengine/README.md), the reusable Python
package, and [allmtest](allmtest/README.md), an example application and test suite.
The default network starts untrained, so correct arithmetic is not guaranteed.

## Installation and quick start

Requires Python 3.10 or newer. Install into the virtual environment used by your
application. From the repository root:

```sh
python -m pip install -e ./allmengine
```

From another directory, replace `./allmengine` with the path to the package
folder. See the [package README](allmengine/README.md) for a compact usage guide.

Installation alone does not change Python's operators. Activate the import hook
once, before importing the application packages you want to instrument:

```python
# launch.py — outside the myapp package
import allmengine

allmengine.enable("myapp")

from myapp.main import main
main()
```

The application stays ordinary Python:

```python
# myapp/calculator.py
def calculate(left, right):
    total = left + right
    if total > 0:
        return total * 2
    return 0
```

Calls such as `calculate(3, 4)` route numeric operations through the shared ALU.
Inputs and results are plain Python values. There is no need for `NeuralValue`
constructors, new function signatures, or changes to application operators.

To select multiple packages, use `allmengine.enable("myapp", "myotherapp")`.
Choose application code only; keep custom models outside those packages.

## Run without a launcher file

Instead of `python -m myapp.main`, run:

```sh
python -m allmengine myapp.main --your-application-argument
```

The runner instruments the entry module's top-level package (`myapp` here).
Explicit selections go before the module name:

```sh
python -m allmengine --package myapp --package sharedlogic myapp.main
```

The runner accepts importable module names, not filesystem script paths. For
an existing standalone `script.py`, run `python -m allmengine script` from its
directory. Putting `enable()` inside an already-running script does not rewrite
that script; use the runner or a separate launcher.

## What activation changes

The hook transforms selected Python source modules in memory before compiling
them. It leaves source files untouched, retains source filenames and line
numbers, and does not reuse or write ordinary bytecode caches for those modules.

Supported syntax includes:

- Binary arithmetic and bitwise operators, including `+`, `-`, `*`, `/`, `%`,
  `**`, shifts, `&`, `|`, and `^`.
- Unary `+`, `-`, `~`, and `not`.
- Numeric comparisons, including chained comparisons.
- Truth checks in `if`, `while`, conditional expressions, `assert`, comprehension
  filters, and pattern-matching guards.
- Short-circuit `and` and `or`, preserving the selected operand as the result.
- Augmented assignments such as `+=`, including attributes and subscripts.

Operands are evaluated in Python order. Chained comparisons evaluate their
middle operands once, and short-circuit expressions skip unneeded operands.
Assignments, imports, iteration, and other Python machinery remain native.

The dispatch layer sends built-in numeric operands to the ALU. Floats are sent
to the model too, so the current integer codec explicitly rejects them.
Operations involving strings, containers, and custom classes retain native
Python behavior. This fallback does not add network data types.

## Why the engine does not call itself recursively

Instrumentation is scoped to explicitly selected application module names.
`allmengine`, pytest infrastructure, and standard-library modules cannot be
selected. Imports of dependencies outside the selected names remain native.
Network weight calculations, bit encoding, and result decoding therefore use
ordinary Python operators, even when triggered by an instrumented operation.

Do not define a model's `forward()` implementation inside an instrumented
application package: its arithmetic would be instrumented too. Keep models in
a separate, unselected module.

## Model and test integration

The default model is the configured, untrained `BinaryNetwork`. Enabling
instrumentation does not make its outputs correct. To verify integration with
native results first, explicitly install the reference model in your launcher:

```python
import allmengine
from allmengine.reference import ReferenceModel

allmengine.alu.set_model(ReferenceModel())  # Wiring check only; not a neural model.
allmengine.enable("myapp")

from myapp.main import main
main()
```

## Current boundaries

- Activation must happen before selected modules are imported. Late activation
  raises an error rather than silently leaving cached code native.
- Only ordinary Python source modules are supported. Extension modules,
  sourceless bytecode, and zip/custom import loaders are not instrumented.
- Built-in and library function internals stay native: `sum(values)`, `bool(x)`,
  `abs(x)`, `operator.add(a, b)`, and NumPy operations are not rewritten as ALU
  calls. Numeric operator syntax in selected code is the interception boundary.
- Identity (`is`) and membership (`in`) are native. A comparison chain mixing
  either with numeric comparisons currently stays native as a whole.
- Pattern matching remains native, except for explicit case guards.
- Dynamically compiled `eval`/`exec` code and functions defined in unselected
  modules remain native.
- This is source instrumentation, not a replacement Python interpreter. It
  introduces an internal runtime binding in each transformed module. Code that
  dynamically changes module globals or depends on exact bytecode may need
  adaptation.

`enable()` returns a handle with `close()` and context-manager support. Closing
it stops instrumentation of future imports; already-loaded functions remain
instrumented. Start a fresh process to return the application fully to normal.

## Network and encoding

The default `BinaryNetwork` has **40 input bits, three hidden layers of 128 bits,
and 16 output bits**. Dimensions, initialization seed, mutation rate, and
mutation scale live in [network.json](allmengine/src/allmengine/network.json).
Hidden and output activations are binary; weights and biases are floating-point.
These widths count logical bits, not Python memory usage.

Set `ALLMENGINE_NETWORK_CONFIG` to an absolute JSON file path before starting
Python, or construct `BinaryNetwork("path/to/network.json")` and install it with
`allmengine.alu.set_model(model)`. Configuration is read when a model is created;
editing the file does not resize an existing model. Checkpoints contain both
configuration and weights.

The current codec accepts bools and signed integers, with 16-bit operand limits
under the default configuration. It cannot represent floats, strings, arbitrary
large integers, or exceptions such as division by zero. The example reference
suite includes cases beyond those limits and is not currently a reachable
training target for this codec.

[BIT_ENCODING.md](BIT_ENCODING.md) specifies operand fields, operation keys,
integer results, and the implemented majority-vote boolean format: more than
half the output bits must be 1 for true; ties are false. The threshold follows
the configured output width. The encoder still puts the opcode first, as noted
in the specification's implementation alignment section.

Network checkpoints store configuration and weights without versioning.
Boolean results always use majority decoding, including for older checkpoints.

The central boundary is `NetworkModel.forward(opcode, lhs, rhs)`, exposed through
`allmengine.alu`. A model returns raw results; BOOL must return a Python `bool`.
For explicit experiments, `NeuralValue` remains available: its arithmetic and
comparison results stay wrapped, and truthiness calls the ALU. Normal application
integration uses the import hook instead. Three-argument `pow` and `divmod` are
not supported by the wrapper interface.

## Training against an external project

Run the configured network against an existing pytest suite:

```sh
python -m allmengine.trainer /path/to/project --factory allmengine.network:BinaryNetwork --instrument myapp --max-evaluations 100 --checkpoint weights.json
```

Repeat `--instrument` for additional application packages. Select application
modules, not the test suite: expected values and assertions should remain native
so an incorrect neural comparison cannot make a broken result pass. The suite
must not replace the installed candidate model.

Each candidate runs in a fresh Python process, with checkpoint state loaded and
instrumentation enabled before pytest imports application code. The trainer
retains non-decreasing passed-test fractions, restores the best state, and stops
when pytest succeeds with at least one passed test or the evaluation budget runs
out. Skips do not count as passes. Collection, setup, teardown, and timeout errors
abort training.

`BinaryNetwork.mutate` perturbs shared weights and biases. This is black-box
mutation search, not gradient-based learning; convergence is not guaranteed.

To use another model, provide an importable zero-argument factory such as
`my_model:Model`, with `forward`, `state_dict`, `load_state_dict`, and
`mutate(random.Random)`. State must be a JSON-compatible dictionary. Keep that
model module outside the instrumented application packages.

## Development and verification

Run the runtime, network, trainer, and example tests from the repository root:

```sh
python -m pytest -q
```

Using the existing Windows virtual environment directly:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

`allmtest/target.py` contains ordinary application logic. Its example tests wrap
inputs explicitly and install `ReferenceModel` for standalone runs, while
trainer workers use their candidate model. Reference tests also inspect raw
outputs so learned equality cannot hide arithmetic errors. The engine tests
separately verify automatic instrumentation of ordinary application code.

After installation, smoke-test the worker against the native reference backend:

```sh
python -m allmengine.trainer ./allmtest --factory allmengine.reference:ReferenceModel --max-evaluations 1
```

Passing reference tests verifies wiring; it does not demonstrate learned neural
arithmetic.
