# allm

Bored with coding?
Your codebase is super clean, well tested and you understand everything?
Make it interesting again, use allm and let the chaos reign.

Let allm wrap over your codebase and let it train on your testcases (lets see how good you tested).
Allm is an experimental architecture for routing Python standard operations through a 
Universal Neural ALU.
So far the ALU takes care of all your ifs, logical operators (AND, OR, ..), basic Arithmetics, mod, ...

If you dont have a codebase with a big enough testset, 
use the attached allmtest, or the distributed weights to let the chaos reign.

Run the wiring and trainer integration tests from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

For a fresh environment, install the engine (which includes pytest):

```sh
python -m pip install -e ./allmengine
python -m pytest -q
```

`allmengine/src/allmengine/cpu.py` defines the process-wide singleton `alu` and
the `NetworkModel.forward(opcode, lhs, rhs)` boundary. Its default model raises
`NotImplementedError`; it never silently computes native arithmetic. Install
your model with `alu.set_model(model)`. Encoding and decoding belong inside the
model. The BOOL opcode must decode to an actual Python `bool`.

`NeuralValue` delegates binary arithmetic, reflected arithmetic, comparisons,
unary operations, and bitwise operations to that same model. Results remain
wrapped, including comparisons, so `if left < right` sends LT and then BOOL.
Python `and`, `or`, and `not` use this truthiness hook and retain Python's
short-circuit behavior. In-place arithmetic uses Python's binary fallback.
Three-argument `pow` and `divmod` are not supported by this initial interface.
Plain unwrapped Python values are unaffected; this is not an interpreter rewrite.

`allmtest/target.py` contains ordinary application logic. Tests wrap inputs at
the boundary. The test suite explicitly installs `ReferenceModel`, a native
operator oracle used only to verify wiring, unless a trainer worker has already
installed a candidate. This backend is not a neural network or a learned model.
Tests also observe raw outputs so learned equality cannot hide arithmetic errors.

## Trainer

Implement a zero-argument model factory in an importable module, providing
`forward`, `state_dict`, `load_state_dict`, and `mutate(random.Random)`.
State is a JSON-compatible dictionary containing the central network's weights;
the model decides how to represent and mutate them. The trainer has no fixed
tensor or layer assumptions.

```sh
python -m allmengine.trainer ./allmtest --factory my_model:Model --max-evaluations 100 --checkpoint weights.json
```

Each candidate runs in a fresh Python process, with its state loaded before
pytest collection. An external suite must wrap its inputs and must not replace
the installed candidate. The trainer retains non-decreasing passed-test fractions,
restores the best state, and stops when pytest succeeds with at least one passed
test or the evaluation budget runs out. Skips do not count as passes. Collection,
setup, teardown, and timeout errors abort training. There is no guarantee of
convergence; real network training remains future work.

After installation, smoke-test the worker against the reference backend:

```sh
python -m allmengine.trainer ./allmtest --factory allmengine.reference:ReferenceModel --max-evaluations 1
```
