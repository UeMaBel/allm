# Use allmengine with ordinary application code

Install the package in the same environment as your application. From this
repository, use `python -m pip install -e ./allmengine`; from another directory,
replace `./allmengine` with the path to the package directory.

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

The trainer can activate instrumentation in each fresh pytest worker before
application modules are imported:

```sh
python -m allmengine.trainer /path/to/project --factory allmengine.network:BinaryNetwork --instrument myapp --max-evaluations 100
```

Repeat `--instrument` for additional application packages. Select application
modules, not the test suite: expected results and test assertions should stay
native so an incorrect neural comparison cannot make a broken result pass.

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
