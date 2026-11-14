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
