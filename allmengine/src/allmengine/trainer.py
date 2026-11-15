"""Black-box training with fresh pytest workers and JSON model checkpoints.

The parent creates and mutates a model. Each evaluation starts a fresh Python
process, loads that candidate's weights, and runs the selected pytest suite.
Reference-based operation fitness decides which candidate to keep; no gradients are used.
The worker writes collection progress and final counts back to the parent.
"""
import argparse
from dataclasses import dataclass, replace
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
from .fitness import RecordingModel, evaluate_operations, operation_key


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
    operations: tuple = ()
    operation_score: float = 0.0
    scored_operations: int = 0
    excluded_operations: tuple = ()

    @property
    def success(self):
        return self.exit_code == 0 and self.passed > 0

    @property
    def test_pass_rate(self):
        # Skipped tests supply no evidence of correctness, so they lower the score.
        total = self.passed + self.failed + self.skipped
        return self.passed / total if total else 0.0


    @property
    def score(self):
        """Mutation fitness; pytest pass rate is reported separately."""
        return self.operation_score


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
            operations = tuple(counts["operations"])
            feedback = Feedback(counts["passed"], counts["failed"], counts["skipped"],
                                process.returncode, output, operations)
            corpus = {operation_key(op): op for op in operations}
            return self._score_operations(model, feedback, corpus.values())

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
        corpus = {operation_key(op): op for op in best.operations}
        best_state = json.loads(json.dumps(model.state_dict()))
        evaluations = 1
        while not best.success and evaluations < max_evaluations:
            # Always mutate the best retained state, not a rejected candidate.
            model.load_state_dict(json.loads(json.dumps(best_state)))
            model.mutate(self.rng)
            print(f"Evaluation {evaluations + 1}/{max_evaluations}", flush=True)
            candidate = self.evaluate(model)
            candidate_state = json.loads(json.dumps(model.state_dict()))
            # A mutation can reach new branches. Add their calls to a shared
            # corpus and rescore BOTH states on it before comparing fitness.
            previous_count = len(corpus)
            corpus.update((operation_key(op), op) for op in candidate.operations)
            if len(corpus) > previous_count:
                print(f"  Shared operation corpus expanded: {previous_count} -> {len(corpus)} calls; "
                      "rescoring both models", flush=True)
            candidate = self._score_operations(model, candidate, corpus.values())
            model.load_state_dict(best_state)
            best = self._score_operations(model, best, corpus.values())
            self._print_feedback(candidate)
            print(f"  Incumbent fitness on same inputs: {best.score:.4%}", flush=True)
            evaluations += 1
            # Ties are kept too, allowing exploration without losing score.
            if candidate.score >= best.score:
                best = candidate
                best_state = candidate_state
                print("  Candidate kept", flush=True)
            else:
                print("  Candidate rejected", flush=True)
        model.load_state_dict(best_state)
        print(f"Training finished: {'converged' if best.success else 'evaluation budget exhausted'}; "
              f"{evaluations} evaluations; retained operation fitness: {best.score:.4%}", flush=True)
        return TrainingResult(best.success, evaluations, best, best_state)

    @staticmethod
    def _score_operations(model, feedback, operations):
        fitness = evaluate_operations(model, operations)
        return replace(feedback, operation_score=fitness.score,
                       scored_operations=fitness.scored,
                       excluded_operations=tuple(fitness.excluded.items()))

    @staticmethod
    def _print_feedback(feedback):
        print(f"  Passed: {feedback.passed}; failed: {feedback.failed}; "
              f"skipped: {feedback.skipped}; pytest pass rate: {feedback.test_pass_rate:.1%}", flush=True)
        print(f"  Operation fitness: {feedback.score:.4%}; "
              f"scored operations: {feedback.scored_operations}", flush=True)
        if not feedback.scored_operations:
            print("  No supported neural operations available for fitness", flush=True)
        for reason, count in feedback.excluded_operations:
            print(f"  Excluded operations ({reason}): {count}", flush=True)


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
        recorder = RecordingModel(model)
        alu.set_model(recorder)
        from .instrumentation import enable
        packages = json.loads(os.environ.get("ALLMENGINE_INSTRUMENT", "[]"))
        hook = enable(*packages) if packages else None
        report = _Report()
        try:
            code = pytest.main(args, plugins=[report])
        finally:
            if hook:
                hook.close()
        report.counts["operations"] = recorder.operations
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
