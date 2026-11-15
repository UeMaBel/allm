"""Black-box training with fresh pytest workers and JSON model checkpoints.

The parent creates and mutates a model. Each evaluation starts a fresh Python
process, loads that candidate's weights, and runs the selected pytest suite.
Passed-test fractions decide which candidate to keep; no gradients are used.
The worker writes collection progress and final counts back to the parent.
"""
import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Protocol

from .cpu import NetworkModel, alu


class TrainableModel(NetworkModel, Protocol):
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def mutate(self, rng: random.Random) -> None:
        """Mutate shared network weights in place."""
        ...


@dataclass(frozen=True)
class Feedback:
    """Results of one candidate; test failures are feedback, worker errors are not."""
    passed: int
    failed: int
    skipped: int
    exit_code: int
    output: str

    @property
    def success(self):
        return self.exit_code == 0 and self.passed > 0

    @property
    def score(self):
        # Skipped tests supply no evidence of correctness, so they lower the score.
        total = self.passed + self.failed + self.skipped
        return self.passed / total if total else 0.0


@dataclass(frozen=True)
class TrainingResult:
    """Best model state and its feedback when training stops."""
    converged: bool
    evaluations: int
    feedback: Feedback
    state: dict


def _create_model(factory):
    """Resolve module:Class and call its zero-argument constructor."""
    module, name = factory.split(":", 1)
    return getattr(importlib.import_module(module), name)()


class Trainer:
    """Keep improvements, revert regressions, stop on success or budget exhaustion.

    factory names an importable zero-argument model constructor (module:name).
    Its state must be JSON serializable. Neural architecture and mutation strategy
    belong to the model, not to this harness.
    """

    def __init__(self, factory, project, pytest_args=(), timeout=60, seed=None, instrument=()):
        self.factory = factory
        self.project = Path(project).resolve()
        self.pytest_args = tuple(pytest_args)
        self.timeout = timeout
        self.seed = seed
        self.rng = random.Random(seed)
        self.instrument = tuple(instrument)

    def evaluate(self, model):
        """Run a candidate in isolation, keeping pytest output for error diagnostics."""
        with tempfile.TemporaryDirectory(prefix="allm-") as directory:
            # JSON carries weights and results across the process boundary.
            state = Path(directory) / "state.json"
            report = Path(directory) / "report.json"
            state.write_text(json.dumps(model.state_dict()), encoding="utf-8")
            env = os.environ.copy()
            progress = Path(directory) / "progress.json"
            env["ALLMENGINE_TRAINER_PROGRESS"] = str(progress)
            env["ALLMENGINE_TRAINER_WORKER"] = "1"
            env["ALLMENGINE_INSTRUMENT"] = json.dumps(self.instrument)
            # Carry source checkouts and custom model modules into fresh workers.
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(p).resolve()) for p in sys.path if p]
                + [str(self.project), env.get("PYTHONPATH", "")]
            )
            command = [sys.executable, "-m", "allmengine.trainer", "--worker",
                       self.factory, str(state), str(report), *self.pytest_args]
            # communicate() drains both pipes while we wait, avoiding a deadlock
            # when pytest produces more output than an OS pipe can hold.
            started = time.monotonic()
            terminal = sys.stdout.isatty()
            collected = None
            tick = 0
            next_update = 0.0
            print("  Loading pytest and collecting tests...", flush=True)
            with subprocess.Popen(command, cwd=self.project, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) as process:
                try:
                    while True:
                        elapsed = time.monotonic() - started
                        if progress.exists() and collected is None:
                            collected = json.loads(progress.read_text(encoding="utf-8"))["collected"]
                            print(("\r" if terminal else "") + f"  Test count: {collected} selected tests" +
                                  (" " * 30 if terminal else ""), flush=True)
                        if elapsed >= next_update:
                            phase = "Running tests" if collected is not None else "Loading / collecting"
                            if terminal:
                                frame = "|/-\\"[tick % 4]
                                print(f"\r  {frame} {phase} ({elapsed:.1f}s)   ", end="", flush=True)
                            elif elapsed >= 5:
                                print(f"  {phase}... {elapsed:.0f}s elapsed", flush=True)
                            tick += 1
                            next_update = elapsed + (0.2 if terminal else 5)
                        remaining = self.timeout - elapsed
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, self.timeout)
                        try:
                            stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                except BaseException as error:
                    # A timed-out/interrupted worker must finish before its
                    # temporary files are removed and before training can exit.
                    process.kill()
                    process.communicate()
                    if isinstance(error, subprocess.TimeoutExpired):
                        raise RuntimeError("pytest evaluation timed out") from error
                    raise
                finally:
                    if terminal:
                        print("\r" + " " * 80 + "\r", end="", flush=True)
            if collected is None and progress.exists():
                collected = json.loads(progress.read_text(encoding="utf-8"))["collected"]
                print(f"  Test count: {collected} selected tests", flush=True)
            output = stdout + stderr
            if process.returncode not in (0, 1) or not report.exists():
                raise RuntimeError(f"pytest infrastructure failure ({process.returncode}):\n{output}")
            counts = json.loads(report.read_text(encoding="utf-8"))
            if counts["errors"]:
                raise RuntimeError(f"pytest setup/teardown failure:\n{output}")
            return Feedback(counts["passed"], counts["failed"], counts["skipped"],
                            process.returncode, output)

    def train(self, max_evaluations=100):
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be at least one")
        print(f"Starting training: {self.factory}", flush=True)
        print(f"  Project: {self.project}", flush=True)
        print(f"  Pytest arguments: {self.pytest_args or '(default discovery)'}", flush=True)
        print(f"  Instrumented packages: {', '.join(self.instrument) or '(none)'}", flush=True)
        print(f"  Evaluation budget: {max_evaluations}; timeout: {self.timeout}s; seed: {self.seed}", flush=True)
        model = _create_model(self.factory)
        training_config = getattr(model, "training_config", None)
        if training_config is not None:
            print(f"  Mutation rate: {training_config.mutation_rate}; "
                  f"scale: {training_config.mutation_scale}", flush=True)
        alu.set_model(model)
        # Evaluate the initial model before making any mutations.
        print(f"Evaluation 1/{max_evaluations} (initial model)", flush=True)
        best = self.evaluate(model)
        self._print_feedback(best)
        best_state = json.loads(json.dumps(model.state_dict()))
        evaluations = 1
        while not best.success and evaluations < max_evaluations:
            # Always mutate the best retained state, not a rejected candidate.
            model.load_state_dict(json.loads(json.dumps(best_state)))
            model.mutate(self.rng)
            print(f"Evaluation {evaluations + 1}/{max_evaluations}", flush=True)
            candidate = self.evaluate(model)
            self._print_feedback(candidate)
            evaluations += 1
            # Ties are kept too, allowing exploration without losing score.
            if candidate.success or candidate.score >= best.score:
                best = candidate
                best_state = json.loads(json.dumps(model.state_dict()))
        model.load_state_dict(best_state)
        print(f"Training finished: {'converged' if best.success else 'evaluation budget exhausted'}; "
              f"{evaluations} evaluations", flush=True)
        return TrainingResult(best.success, evaluations, best, best_state)

    @staticmethod
    def _print_feedback(feedback):
        print(f"  Passed: {feedback.passed}; failed: {feedback.failed}; "
              f"skipped: {feedback.skipped}; score: {feedback.score:.1%}", flush=True)


class _Report:
    """Pytest plugin running inside the worker, separate from the parent trainer."""
    def __init__(self):
        self.counts = dict(passed=0, failed=0, skipped=0, errors=0)

    def pytest_collection_finish(self, session):
        # This hook runs after selection (-k, -m, etc.). Publish atomically so
        # the parent never tries to read a half-written JSON document.
        progress_path = os.environ.get("ALLMENGINE_TRAINER_PROGRESS")
        if progress_path:
            path = Path(progress_path)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"collected": len(session.items)}), encoding="utf-8")
            os.replace(temporary, path)

    def pytest_runtest_logreport(self, report):
        # Count test bodies as feedback; setup/teardown failures indicate a
        # broken evaluation environment and must not become model scores.
        if report.when == "call":
            self.counts[report.outcome] += 1
        elif report.failed:
            self.counts["errors"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1


def main():
    # The same module is the user CLI and the private worker entry point.
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        import pytest
        factory, state_path, report_path, *args = sys.argv[2:]
        model = _create_model(factory)
        model.load_state_dict(json.loads(Path(state_path).read_text(encoding="utf-8")))
        alu.set_model(model)
        from .instrumentation import enable
        packages = json.loads(os.environ.get("ALLMENGINE_INSTRUMENT", "[]"))
        hook = enable(*packages) if packages else None
        report = _Report()
        try:
            code = pytest.main(args, plugins=[report])
        finally:
            if hook:
                hook.close()
        Path(report_path).write_text(json.dumps(report.counts), encoding="utf-8")
        return int(code)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--factory", required=True, help="module:ModelClass")
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--instrument", action="append", default=[], metavar="PACKAGE",
                        help="Instrument an application package before pytest imports it (repeatable)")
    args = parser.parse_args()
    result = Trainer(args.factory, args.project, timeout=args.timeout,
                     seed=args.seed, instrument=args.instrument).train(args.max_evaluations)
    if args.checkpoint:
        args.checkpoint.write_text(json.dumps(result.state, indent=2), encoding="utf-8")
    print(f"converged={result.converged} evaluations={result.evaluations} "
          f"passed={result.feedback.passed} failed={result.feedback.failed}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
