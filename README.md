# allm

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](allmengine/pyproject.toml)
[![Tests: pytest](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](#development-and-verification)
[![Status: experimental](https://img.shields.io/badge/Status-experimental-orange)](#current-boundaries)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen)](LICENSE)

Bored with coding? Your codebase is clean, your tests are green, and addition
stubbornly keeps returning the right answer? Give your operators to an untrained
neural network. Suddenly, `2 + 2` is a matter of opinion, `if` statements have
trust issues, and your test suite has a new job: teaching Python math again.
Use allm. Let the chaos reign.

allm is an experimental project that routes selected Python operations through
one shared neural ALU, then learns its weights from reference targets for operations observed in your tests.
Application code keeps ordinary operators; one-time activation selects the
packages to instrument.

The repository contains [allmengine](allmengine/README.md), the reusable Python
package, and [allmtest](allmtest/README.md), an example application and test suite.
The default network starts untrained, so correct arithmetic is not guaranteed.

This project still has plenty of rough edges. Expect incomplete coverage,
surprising behavior, and changes to the interfaces as the experiment evolves.
It is a playground for experiments, not something to rely on in production.

## Contents

- [Installation and quick start](#installation-and-quick-start)
- [What activation changes](#what-activation-changes)
- [Model and test integration](#model-and-test-integration)
- [Current boundaries](#current-boundaries)
- [Future idea: a dedicated Python interpreter](#future-idea-a-dedicated-python-interpreter)
- [Network and encoding](#network-and-encoding)
- [Training against an external project](#training-against-an-external-project)
- [Development and verification](#development-and-verification)
- [Three ways to use allmengine](#three-ways-to-use-allmengine)
- [License](#license)

## Installation and quick start

Requires Python 3.10 or newer. Install into the virtual environment used by your
application. From the repository root:

```sh
python -m pip install -e ./allmengine
```

From another directory, replace `./allmengine` with the path to the package
folder. See the [package README](allmengine/README.md) for a compact usage guide.

### Run without a launcher file

Run an existing Python script and its arguments directly:

```sh
uv run python -m allmengine --run "manage.py runserver" --package myapp
```

Replace `myapp` with the application package containing your logic. Repeat
`--package` to select additional packages. The entry script is instrumented too.

For an importable module, use:

```sh
uv run python -m allmengine --run "-m myapp.main --your-application-argument"
```

The module's top-level package (`myapp` here) is selected by default. Quote the
entry point and all its arguments together. Use `python -m allmengine` without
`uv run` if you manage your virtual environment another way.

The application keeps ordinary Python operators and plain inputs and results.
No launcher file or `NeuralValue` wrappers are required. The default network
starts with random weights; add `--weights weights.json` to load trained weights.

For training and integration through `import allmengine`, see
[Three ways to use allmengine](#three-ways-to-use-allmengine).

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

The dispatch layer sends built-in numeric operands to the ALU. If either
operand is a built-in float, the ALU uses native Python instead of the network.
This covers mixed int/float arithmetic, comparisons, unary operations, and
float truthiness, including `NeuralValue` use. For example, `2 + 0.5` returns
`2.5`. Native errors such as float division by zero still apply. Operations
with only bool/int operands still use the network; `3 / 2` remains neural,
whereas `3 / 2.0` is native. A boolean produced by a float comparison is a
normal bool, so a later truth check on it follows the usual bool ALU path.
Operations involving strings, containers, and custom classes retain native
Python behavior. This fallback does not add network data types.

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

## Future idea: a dedicated Python interpreter

A future idea is to build a new Python interpreter that routes operations
through a shared, trainable neural ALU directly during execution. That could
extend the experiment beyond the current source instrumentation and make neural
operations part of the interpreter itself. This is an exploratory idea; the
current implementation still uses Python's existing interpreter and import hooks.

## Network and encoding

The default `BinaryNetwork` has **40 input bits, three hidden layers of 128 ReLU units,
and 16 output bits**. Dimensions and initialization seed live in
[network.json](allmengine/src/allmengine/network.json). Adam learning rate,
batch size, epochs between evaluations, and shuffling live in
[training.json](allmengine/src/allmengine/training.json).
PyTorch computes continuous hidden activations and linear output scores.
Runtime thresholds output scores at zero; training uses binary cross-entropy
with logits before thresholding. The historical `hidden_bits` key counts units.
Input encoding and integer/majority-boolean decoding remain unchanged.

Set `ALLMENGINE_NETWORK_CONFIG` to an absolute JSON file path before starting
Python, or construct `BinaryNetwork("path/to/network.json")` and install it with
`allmengine.alu.set_model(model)`. Configuration is read when a model is created;
editing the file does not resize an existing model. Checkpoints contain both
configuration and weights.

The current codec accepts bools and signed integers, with 16-bit operand limits
under the default configuration. It cannot represent floats, strings, arbitrary
large integers, or exceptions such as division by zero. Runtime float operations
bypass the codec and run natively; direct `BinaryNetwork.forward` and `encode`
calls still reject floats. The example reference suite includes large integers
and integer division-by-zero cases beyond the network's capabilities, so it is
not currently a reachable training target for this codec.

[BIT_ENCODING.md](BIT_ENCODING.md) specifies operand fields, operation keys,
integer results, and the implemented majority-vote boolean format: more than
half the output bits must be 1 for true; ties are false. The threshold follows
the configured output width. The input layout is **8-bit opcode → 16-bit left
operand → 16-bit right operand**, matching the implementation and specification.

Network checkpoints store configuration, weights, and the explicit model type
`relu-bit-network-v1`. Old binary-activation checkpoints are incompatible and
require retraining. Existing `weights.json` and `pretrained.json` files are not
converted or overwritten automatically.

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
retains the best exact operation accuracy (breaking ties by operation fitness),
restores that state, and stops
when pytest succeeds with at least one passed test or the evaluation budget runs
out. Skips do not count as passes. Collection, setup, teardown, and timeout errors
abort training.

`BinaryNetwork` uses Adam and uniform per-bit binary cross-entropy on reference
targets for the observed operations. Newly reached branches expand the training
corpus; both current and retained weights are scored on the same inputs.
Training updates continue from the latest weights, independently of which
checkpoint is retained. Passing pytest takes precedence over operation metrics.
This trains individual ALU calls, not gradients through Python control flow.
Convergence is not guaranteed.

Both training CLIs accept `--learning-rate`, `--batch-size`, `--epochs`,
`--shuffle` / `--no-shuffle`, `--training-config PATH`,
`--time-budget SECONDS`, and `--checkpoint-interval SECONDS`. `--epochs` counts
passes between evaluations; the first evaluation makes no updates. Individual
flags override an explicit settings file, which overrides checkpoint settings.
Without a checkpoint, settings come from `ALLMENGINE_TRAINING_CONFIG` or the
packaged `training.json`. Checkpoints restore model/settings only, not Adam state.

For an unattended run, use `--time-budget 3300 --checkpoint-interval 60` and a
large evaluation cap. This gives a cooperative 55-minute budget, saving the best
evaluated weights about every minute and on exit, including Ctrl+C. An active
evaluation cycle can overrun the limit. Saves atomically replace `--checkpoint`;
updates since the last evaluation are discarded if time expires mid-training.
A budget stop preserves a usable checkpoint but exits with code 1 (not converged).
See [timed runs](allmengine/README.md#timed-runs-and-periodic-checkpoints) for a
complete command and the timing details.
See [the package guide](allmengine/README.md#gradient-pretraining-example) for
pretraining and fine-tuning examples.

To use another model, provide an importable zero-argument factory such as
`my_model:Model`, with `forward`, `state_dict`, `load_state_dict`, and
`mutate(random.Random)`. State must be a JSON-compatible dictionary. Keep that
model module outside the instrumented application packages. Custom models that
are not `BinaryNetwork` subclasses continue using mutation search.

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

## Three ways to use allmengine

Run these commands from the repository root after installing the engine into
your environment (`python -m pip install -e ./allmengine`). Use `python -m allmengine`,
not `python allmengine`: `-m` runs the installed Python package.

### 1. Run an application

```powershell
uv run python -m allmengine --run "manage.py runserver" --package myapp
uv run python -m allmengine --run "-m myapp.main --port 8000" --weights weights.json
```

Replace `myapp` with the application package containing your logic. Repeat
`--package` for additional packages. The script itself is instrumented too.
For module entry points, the default selection is the module's top-level package;
for scripts it is the script's stem. Select your application packages explicitly
when a launcher such as `manage.py` imports the actual application.

Without `--weights`, execution uses a new random network. With `--weights`, it
loads the saved model state. Quote the entire Python command and its arguments.
These commands run Python code, not shell pipelines or shell commands.

### 2. Train against tests

```powershell
uv run python -m allmengine --train "pytest allmtest/tests -q" --max-evaluations 100 --checkpoint weights.json
```

Training uses reference targets from pytest-observed calls, backpropagates with
Adam, and saves the best evaluated state.
The quoted command can include pytest paths, filters, and other pytest options.
For ordinary application code, add `--package myapp` to instrument it before
pytest imports it. The allmtest suite already uses `NeuralValue` wrappers, so it
needs no package selection. Keep test assertions outside instrumented packages.

Normal pytest runs in allmtest install the native reference backend; training
bypasses that override automatically. Training starts with a new model each time;
`--checkpoint` saves results; `--weights` loads a compatible ReLU checkpoint.
Gradient training may not converge. Large integers and integer division-by-zero cases in the
current suite exceed the codec's capabilities; float operations run natively.

### 3. Integrate in Python code

Put activation in a separate launcher, before importing the application:

```python
import json
from pathlib import Path

import allmengine

# Optional: load trained weights. Omit this block to use random weights.
model = allmengine.BinaryNetwork()
model.load_state_dict(json.loads(Path("weights.json").read_text(encoding="utf-8")))
allmengine.alu.set_model(model)

allmengine.enable("myapp")
from myapp.main import main

main()
```

Application code keeps normal operators such as `+`, `*`, and comparisons.
Calling `enable()` does not rewrite the launcher that is already running.
For a deterministic integration check, explicitly install
`allmengine.reference.ReferenceModel()` instead of the neural model.

## License

Licensed under the [WTFPL, version 2](LICENSE) (Do What The Fuck You Want To
Public License). Do what you want with it.
