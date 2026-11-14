"""Run an application module with selected packages instrumented."""
import argparse
import runpy
import sys

from .instrumentation import enable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", default=[],
                        help="Application package to instrument (repeatable; defaults to module's top-level package)")
    parser.add_argument("module", help="Module to run, as with python -m")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
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
    main()
