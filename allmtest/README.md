# allmtest

A pytest suite for arithmetic and control flow using `NeuralValue` inputs.
Standalone runs explicitly install the native reference backend to verify wiring;
trainer workers use the supplied candidate model instead. See the root README
for the model contract and trainer usage.

Install the dependencies and run the suite from this directory:

```sh
uv sync
uv run pytest
```

## Why normal test runs pass

`tests/conftest.py` installs `ReferenceModel` for standalone pytest runs. This
backend uses native Python arithmetic, so passing tests verify the integration,
not the accuracy of random neural weights.

## Train against these test cases

From the repository root, using the existing Windows virtual environment:

```powershell
.venv/Scripts/python.exe -m allmengine.trainer ./allmtest --factory allmengine.network:BinaryNetwork --max-evaluations 100 --checkpoint weights.json
```

Alternatively, from the `allmtest` directory with uv:

```sh
uv sync
uv run python -m allmengine.trainer . --factory allmengine.network:BinaryNetwork --max-evaluations 100 --checkpoint weights.json
```

The trainer automatically bypasses the reference backend. Each candidate runs
against the pytest suite in a fresh worker process. The trainer uses Adam on
reference targets for observed operations, retaining the best exact operation
accuracy (then bit fitness), and writes the retained model state to
`weights.json`, relative to the directory where you ran the command. It stops
when the suite passes or the evaluation budget is exhausted.

Increase `--max-evaluations` for a longer run. `--epochs` controls gradient
training passes between evaluations; `--learning-rate` and `--batch-size` tune
Adam. Improvement and convergence are not guaranteed. `--weights PATH` loads
a compatible ReLU checkpoint; legacy binary-activation weights need retraining.
`--checkpoint` selects the output file, not an input checkpoint.

The tests already wrap inputs in `NeuralValue`, so no `--instrument` flag is
needed. Add arithmetic cases to the `tests/*_cases.py` files or add pytest tests
using `NeuralValue`. Keep assertions on raw results, such as
`assert result.value == expected`, so an incorrect learned equality operation
cannot hide an arithmetic error.

### Choose reachable training cases

The current suite includes cases the network cannot represent, so passing the
entire suite is not currently a reachable training target:

- Use integer inputs and results within the configured codec range. The default
  operands and integer results are signed 16-bit values (-32768 through 32767).
- Integer division-by-zero exceptions cannot be represented by the network.
- Operations involving built-in floats run natively and provide no training
  signal.

For meaningful training, use a suite limited to supported cases. Configuration
lives in `../allmengine/src/allmengine/network.json`.

## Inspect the untrained network directly

To bypass the reference backend without running the trainer, run from the
repository root in PowerShell:

```powershell
$env:ALLMENGINE_TRAINER_WORKER = "1"
try {
    .venv/Scripts/python.exe -m pytest allmtest/tests -q
} finally {
    Remove-Item Env:ALLMENGINE_TRAINER_WORKER
}
```

Expect failures with random weights, although some cases may pass by chance or
because they use native float operations. This command creates a default model;
it does not load the saved `weights.json`.

## Unified interface: run, train, or import

See [the three-mode tutorial](../README.md#three-ways-to-use-allmengine).
From the repository root, training this suite is now:

```powershell
uv run python -m allmengine --train "pytest allmtest/tests -q" --max-evaluations 100 --checkpoint weights.json
```

The older `python -m allmengine.trainer` interface below remains supported.

## Calculator demo

From this directory, compare ordinary Python with the random neural network:

```powershell
uv run python calculator.py 2 + 3
uv run python -m allmengine --run "calculator.py 2 + 3"
```

The first prints `2 + 3 = 5`. The second sends the addition through the neural
network: the answer depends on its weights. Try subtraction, multiplication,
 and division too:

```powershell
uv run python -m allmengine --run "calculator.py 12 - 5"
uv run python -m allmengine --run "calculator.py 6 * 7"
uv run python -m allmengine --run "calculator.py 8 / 2"
```

Load trained weights with `--weights ../weights.json` (adjust the path to your
checkpoint). Use small integers for neural math; decimal inputs deliberately
use native Python arithmetic. For negative operands, put `--` before the values,
for example `calculator.py -- -2 + -3`.
