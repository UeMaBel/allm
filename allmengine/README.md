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
implement arithmetic correctly. It has 40 input bits, three 128-unit hidden
layers, and 16 output bits. PyTorch computes floating-point ReLU hidden
activations and linear output scores; only the output scores are thresholded
at zero into bits. The historical `hidden_bits` setting counts hidden units.

Edit [network.json](src/allmengine/network.json), set
`ALLMENGINE_NETWORK_CONFIG` to an absolute config path before starting Python,
or install a model explicitly:

```python
from allmengine import BinaryNetwork, alu

alu.set_model(BinaryNetwork("path/to/network.json"))
```

Adam settings live separately in [training.json](src/allmengine/training.json).
Set `ALLMENGINE_TRAINING_CONFIG` to an absolute path before starting Python, or
pass `training_config=TrainingConfig(...)` or a JSON path to `BinaryNetwork`.
`TrainingConfig` is exported from `allmengine`.

```json
{
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 10,
  "shuffle": true
}
```

Both training modes accept `--training-config PATH`, `--learning-rate`,
`--batch-size`, `--epochs`, and `--shuffle` / `--no-shuffle`. `epochs` means
passes over the training corpus **between evaluations**. With 100 evaluations
and 10 epochs, at most 990 epochs run: the first evaluation checks the initial
model. `--seed` controls training shuffling and, in pretraining, data generation;
`network.json`'s `seed` controls weight initialization. Training runs on CPU.
For small workloads, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` can avoid CPU
thread scheduling overhead.

Settings precedence: defaults/environment, then checkpoint settings when loading
`--weights`, then an explicit `--training-config` file, then individual CLI flags.
Old mutation settings are no longer valid in `training.json`.

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
Checkpoints store configuration, weights, and `model_type: relu-bit-network-v1`.
Legacy binary-activation checkpoints (including the repository's old
`weights.json` / `pretrained.json`) are rejected with a retraining message.
Their hidden-layer semantics differ; they cannot be silently reused.

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

The trainer learns from reference targets for observed operations using Adam and
uniform per-bit binary cross-entropy with logits. It keeps the best evaluated
checkpoint by exact operation accuracy, then operation fitness, and stops when
pytest passes or the evaluation budget is exhausted. Successful pytest results
take precedence. Gradient updates continue from the latest weights even when
the retained checkpoint is better. Learning does not guarantee convergence. Choose tests whose inputs and
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

For the unified `--run` / `--train` CLI and Python import integration, see
[the three-mode tutorial](../README.md#three-ways-to-use-allmengine).

### Operation fitness

Tests stay unchanged. Training workers record neural opcodes, operands, and raw
output bits. `ReferenceModel` supplies native expected results for fitness only;
application code always receives the candidate's own answers.

Integer arithmetic uses positional bit weights (including the magnitude of the
sign bit). Bitwise AND, OR, XOR, inversion, and shifts use equal bit weights.
Comparisons and truthiness score decoded boolean correctness under majority voting.
Scores are normalized per operation and averaged over distinct opcode/input calls.
Native float fallbacks do not enter this neural-operation corpus.

Newly reached calls expand a shared corpus; the candidate and incumbent are both
rescored on that same corpus before selection. Exact decoded-answer accuracy is
reported alongside bit fitness. These evaluation metrics are separate from the
uniform per-bit BCE training loss. Predicates train toward all-zero or all-one
targets; evaluation accepts any correct majority. Pytest pass rate is printed
separately and remains the convergence check. Operand overflow, result overflow,
non-integral results, and reference exceptions are excluded with visible counts;
perfect operation fitness alone does not imply all tests can pass.
Custom models without a bit interface use exact operation-result correctness
against the reference. Custom non-`BinaryNetwork` factories retain the existing
mutation-search interface; PyTorch gradient training is used for `BinaryNetwork`
and its subclasses. A corpus with no supported operations fails clearly when
a gradient update is requested.

To initialize training from a saved checkpoint:

```sh
python -m allmengine --train "pytest allmtest/tests -q" --weights weights.json --checkpoint weights.json
```

`--weights` loads the model configuration and parameters before the first
evaluation. `--checkpoint` selects where the retained result is saved; use a
different path to preserve the input file. Without `--weights`, training starts
from a newly initialized model. This resumes model state, not the previous
Adam optimizer state, random-generator state, or recorded operation corpus. The standalone
`python -m allmengine.trainer` CLI also accepts `--weights`.

## Training architecture and pretraining

Training implementation lives in `training/`: `config.py` loads settings,
`examples.py` generates reproducible reference examples, `fitness.py` scores
operations. `targets.py` centralizes representability checks and target bits;
`gradient.py` owns Adam and checkpoint selection. `search.py` retains mutation
search for custom black-box models.
`pretrain.py` supplies generated data; `pytest_runner.py` supplies calls observed
in pytest workers. The original `trainer.py`, `fitness.py`, and
`training_config.py` preserve existing imports and CLI usage.

```sh
python -m allmengine --pretrain --examples 100 --validation-examples 50 --max-evaluations 100 --checkpoint pretrained.json
python -m allmengine --train "pytest allmtest/tests -q" --weights pretrained.json
```

Pretraining runs in memory, uses fixed distinct examples with cached reference
results, and reports fitness on a separate held-out set at the end. Configure
operations and operand range using `--operations add sub mul`, `--minimum`, and
`--maximum`. Generation rejects unsupported results and fails clearly if it
cannot fill the requested dataset. Power and matrix multiplication generation
are currently unsupported. `--weights` initializes either training mode.
The checkpoint is written at completion. Adam settings live in
`training.json`; dataset options belong to the pretraining CLI. Validation
examples never enter gradient updates or checkpoint selection. Pretraining
convergence means exact answers on the training corpus, not general arithmetic
correctness.

### Gradient pretraining example

```sh
python -m allmengine --pretrain --operations add sub --examples 500 --validation-examples 100 --minimum -16 --maximum 16 --learning-rate 0.001 --batch-size 64 --epochs 10 --max-evaluations 100 --checkpoint relu-weights.json
python -m allmengine --train "pytest mytests -q" --package myapp --weights relu-weights.json --epochs 5 --checkpoint finetuned.json
```

The network's `forward_logits(tensor)` exposes differentiable scores to the
trainer. Runtime `forward_bits(bits)` uses inference mode and returns Python
integers; `forward(opcode, lhs, rhs)` retains the existing codec and decoder.
The JSON checkpoint interface stays separate from the internal PyTorch module's
tensor state dictionary. Install dependencies again after upgrading:
`python -m pip install -e ./allmengine` from the repository root.
