"""Run an application module with selected packages instrumented."""
import argparse
import runpy
import sys
import shlex
from pathlib import Path

from .instrumentation import enable
from .training.config import add_training_arguments, training_overrides


def main():
    # Pretraining owns its dataset options and uses no pytest worker.
    if "--pretrain" in sys.argv[1:]:
        if "--run" in sys.argv[1:] or "--train" in sys.argv[1:]:
            raise SystemExit("Choose only one of --pretrain, --train, or --run")
        from .training.pretrain import main as pretrain_main
        previous = sys.argv
        try:
            sys.argv = [arg for arg in sys.argv if arg != "--pretrain"]
            return pretrain_main()
        finally:
            sys.argv = previous
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", action="store_true", help="Train on generated equations (use --pretrain --help for options)")
    add_training_arguments(parser)
    parser.add_argument("--seed", type=int, help="Training shuffle seed")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", metavar="COMMAND", help='Run Python code, e.g. "manage.py runserver" or "-m myapp.main"')
    modes.add_argument("--train", metavar="COMMAND", help='Train against pytest, e.g. "pytest allmtest/tests -q"')
    parser.add_argument("--factory", default="allmengine.network:BinaryNetwork")
    parser.add_argument("--weights", type=Path, help="Load model state for --run or initialize --train")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights.json"))
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--package", action="append", default=[],
                        help="Application package to instrument (repeatable; defaults to module's top-level package)")
    parser.add_argument("module", nargs="?", help="Module to run, as with python -m (legacy syntax)")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.run or args.train:
        if args.module:
            parser.error("Put application arguments inside the quoted command")
        command = shlex.split(args.run or args.train)
        if command and Path(command[0]).name.lower() in ("python", "python.exe", "python3"):
            command.pop(0)
        if args.train:
            if command[:2] == ["-m", "pytest"]:
                command = command[2:]
            elif command and command[0] == "pytest":
                command = command[1:]
            else:
                parser.error('--train requires a pytest command, e.g. "pytest allmtest/tests"')
            from .trainer import Trainer
            import json
            result = Trainer(args.factory, Path.cwd(), pytest_args=command,
                             timeout=args.timeout, instrument=args.package, weights=args.weights,
                             seed=args.seed, training_config=args.training_config,
                             training_options=training_overrides(args)).train(args.max_evaluations)
            args.checkpoint.write_text(json.dumps(result.state, indent=2), encoding="utf-8")
            print(f"converged={result.converged} evaluations={result.evaluations} "
                  f"passed={result.feedback.passed} failed={result.feedback.failed} "
                  f"checkpoint={args.checkpoint}")
            return 0 if result.converged else 1
        if not command:
            parser.error("--run requires a Python script or module")
        from .trainer import _create_model
        from .cpu import alu
        import json
        model = _create_model(args.factory)
        if args.weights:
            model.load_state_dict(json.loads(args.weights.read_text(encoding="utf-8")))
        alu.set_model(model)
        if command[0] != "-m":
            script = Path(command[0]).resolve()
            packages = args.package or [script.stem]
            from .instrumentation import _compile
            previous = sys.argv
            previous_path = sys.path[:]
            try:
                sys.argv = [str(script), *command[1:]]
                sys.path.insert(0, str(script.parent))
                with enable(*packages):
                    exec(_compile(script.read_bytes(), str(script)),
                         dict(__name__="__main__", __file__=str(script), __package__=None, __spec__=None))
            finally:
                sys.argv = previous
                sys.path[:] = previous_path
            return 0
        if len(command) < 2:
            parser.error("-m requires a module name")
        args.module, args.args = command[1], command[2:]
    if not args.module:
        parser.error("Choose --run COMMAND or --train COMMAND")
    packages = args.package or [args.module.split(".")[0]]
    if not any(args.module == name or args.module.startswith(name + ".") for name in packages):
        parser.error("The entry module must belong to a selected --package")
    previous = sys.argv
    try:
        sys.argv = [args.module, *args.args]
        with enable(*packages):
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
