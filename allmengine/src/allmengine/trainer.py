"""Black-box training with fresh pytest workers and JSON model checkpoints."""
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
        total = self.passed + self.failed + self.skipped
        return self.passed / total if total else 0.0


@dataclass(frozen=True)
class TrainingResult:
    converged: bool
    evaluations: int
    feedback: Feedback
    state: dict


def _create_model(factory):
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
        self.rng = random.Random(seed)
        self.instrument = tuple(instrument)

    def evaluate(self, model):
        with tempfile.TemporaryDirectory(prefix="allm-") as directory:
            state = Path(directory) / "state.json"
            report = Path(directory) / "report.json"
            state.write_text(json.dumps(model.state_dict()), encoding="utf-8")
            env = os.environ.copy()
            env["ALLMENGINE_TRAINER_WORKER"] = "1"
            env["ALLMENGINE_INSTRUMENT"] = json.dumps(self.instrument)
            # Carry source checkouts and custom model modules into fresh workers.
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(p).resolve()) for p in sys.path if p]
                + [str(self.project), env.get("PYTHONPATH", "")]
            )
            command = [sys.executable, "-m", "allmengine.trainer", "--worker",
                       self.factory, str(state), str(report), *self.pytest_args]
            try:
                process = subprocess.run(command, cwd=self.project, env=env,
                                         capture_output=True, text=True,
                                         timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("pytest evaluation timed out") from error
            output = process.stdout + process.stderr
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
        model = _create_model(self.factory)
        alu.set_model(model)
        best = self.evaluate(model)
        best_state = json.loads(json.dumps(model.state_dict()))
        evaluations = 1
        while not best.success and evaluations < max_evaluations:
            model.load_state_dict(json.loads(json.dumps(best_state)))
            model.mutate(self.rng)
            candidate = self.evaluate(model)
            evaluations += 1
            if candidate.success or candidate.score >= best.score:
                best = candidate
                best_state = json.loads(json.dumps(model.state_dict()))
        model.load_state_dict(best_state)
        return TrainingResult(best.success, evaluations, best, best_state)


class _Report:
    def __init__(self):
        self.counts = dict(passed=0, failed=0, skipped=0, errors=0)

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.counts[report.outcome] += 1
        elif report.failed:
            self.counts["errors"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1


def main():
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
