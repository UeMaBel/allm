import importlib
from pathlib import Path
import subprocess


def test_script_runner_always_enters_main_guard(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text('if __name__ == "__main__":\n    print("entered")\n')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.run(
        [sys.executable, "-m", "allmengine", "--run", f'"{script}"', "--package", "demo"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "entered"
import sys
import os

import pytest

from allmengine import OpCode, alu, enable
from allmengine.instrumentation import _compile
from allmengine.reference import ReferenceModel
from allmengine.trainer import Trainer


class Recorder(ReferenceModel):
    def __init__(self):
        self.calls = []

    def forward(self, opcode, lhs, rhs=None):
        self.calls.append((opcode, lhs, rhs))
        return super().forward(opcode, lhs, rhs)


@pytest.fixture
def model(monkeypatch):
    model = Recorder()
    monkeypatch.setattr(alu, "model", model)
    return model


def execute(source):
    namespace = {}
    exec(_compile(source, "application.py"), namespace)
    return namespace


def test_plain_literals_arguments_and_results_use_alu(model):
    ns = execute('''
def calculate(a, b):
    total = a + b
    if total > 0:
        return total * 2
    return -total
result = calculate(3, 4)
''')
    assert type(ns["result"]) is int
    assert ns["result"] == 14
    assert model.calls == [(OpCode.ADD, 3, 4), (OpCode.GT, 7, 0),
                           (OpCode.BOOL, True, None), (OpCode.MUL, 7, 2)]


def test_float_application_operations_bypass_network(model):
    ns = execute('''
value = 0.5 + 2
value *= 2
ratio = value / 2.0
comparison = ratio > 1.0
negative = -ratio
branch = "false"
if ratio:
    branch = "true"
selected = 0.0 or ratio
''')
    assert ns["value"] == 5.0
    assert ns["ratio"] == 2.5
    assert ns["comparison"] is True
    assert ns["negative"] == -2.5
    assert ns["branch"] == "true"
    assert ns["selected"] == 2.5
    assert model.calls == []


def test_literal_expressions_are_not_constant_folded_away(monkeypatch):
    class WrongModel:
        def forward(self, opcode, lhs, rhs=None):
            return 999
    monkeypatch.setattr(alu, "model", WrongModel())
    assert execute("result = 2 * 3")["result"] == 999


def test_boolean_values_and_short_circuiting(model):
    ns = execute('''
events = []
def operand(n):
    events.append(n)
    return n
first = operand(0) and operand(99)
second = operand(7) or operand(99)
third = operand(1) and operand(2)
if operand(0) and operand(99):
    raise RuntimeError()
fourth = not operand(0)
''')
    assert ns["events"] == [0, 7, 1, 2, 0, 0]
    assert (ns["first"], ns["second"], ns["third"], ns["fourth"]) == (0, 7, 2, True)
    assert model.calls == [(OpCode.BOOL, n, None) for n in [0, 7, 1, 0, 0]]


def test_chained_comparisons_evaluate_middle_once_and_skip_tail(model):
    ns = execute('''
events = []
def operand(n):
    events.append(n)
    return n
result = operand(3) < operand(2) < operand(99)
if operand(1) < operand(2) < operand(3):
    branch = "yes"
''')
    assert ns["events"] == [3, 2, 1, 2, 3]
    assert ns["result"] is False
    assert ns["branch"] == "yes"
    assert model.calls == [(OpCode.LT, 3, 2), (OpCode.BOOL, False, None),
                           (OpCode.LT, 1, 2), (OpCode.BOOL, True, None),
                           (OpCode.LT, 2, 3), (OpCode.BOOL, True, None)]


def test_while_comprehensions_and_ternary(model):
    ns = execute('''
n = 2
while n:
    n -= 1
values = [n + 1 for n in [0, 1, 2] if n]
result = 10 if 0 else 20
assert n == 0
''')
    assert ns["n"] == 0
    assert ns["values"] == [2, 3]
    assert ns["result"] == 20
    assert (OpCode.SUB, 2, 1) in model.calls
    assert (OpCode.BOOL, 0, None) in model.calls


def test_augmented_targets_are_evaluated_before_rhs_once(model):
    ns = execute('''
events = []
class Box:
    def __init__(self):
        self.value = 10
    @property
    def x(self):
        events.append("get")
        return self.value
    @x.setter
    def x(self, value):
        events.append("set")
        self.value = value
box = Box()
def owner():
    events.append("owner")
    return box
def rhs():
    events.append("rhs")
    return 2
owner().x += rhs()
items = [3]
items[0] *= 4
alias = items
items[:] += [5]
''')
    assert ns["events"] == ["owner", "get", "rhs", "set"]
    assert ns["box"].value == 12
    assert ns["items"] == [12, 5]
    assert ns["alias"] is ns["items"]


def test_native_containers_strings_and_identity_keep_behavior(model):
    ns = execute('''
text = "hello" + " world"
items = [1]
alias = items
items += [2]
same = items is alias
member = 2 in items
empty = not []
''')
    assert ns["text"] == "hello world"
    assert ns["same"] and ns["member"] and ns["empty"]
    assert ns["items"] is ns["alias"]
    assert not model.calls


def test_async_and_generator_operand_evaluation(model):
    ns = execute('''
import asyncio
async def value(n):
    return n
async def calculate():
    return await value(2) + await value(3)
def generate():
    return (yield 1) + (yield 2)
result = asyncio.run(calculate())
''')
    assert ns["result"] == 5
    generator = ns["generate"]()
    assert next(generator) == 1
    assert generator.send(3) == 2
    with pytest.raises(StopIteration) as stopped:
        generator.send(4)
    assert stopped.value.value == 7


def test_import_hook_instruments_only_selected_package(tmp_path, monkeypatch, model):
    package = tmp_path / "selected_app"
    package.mkdir()
    (package / "__init__.py").write_text('"""Application docs."""\nfrom __future__ import annotations\n')
    (package / "logic.py").write_text("def calculate(a, b):\n    return a * b\n")
    (tmp_path / "native_dependency.py").write_text("def calculate(a, b):\n    return a * b\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with enable("selected_app"):
            app = importlib.import_module("selected_app.logic")
            dependency = importlib.import_module("native_dependency")
            assert app.calculate(3, 4) == 12
            assert dependency.calculate(5, 6) == 30
            assert model.calls == [(OpCode.MUL, 3, 4)]
            assert sys.modules["selected_app"].__doc__ == "Application docs."
    finally:
        for name in ("selected_app", "selected_app.logic", "native_dependency"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("name", ["allmengine", "allmengine.network", "operator", "pytest", "", "a-b"])
def test_unsafe_or_invalid_selection_is_rejected(name):
    with pytest.raises(ValueError):
        enable(name)


def test_already_imported_application_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "loaded_app.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        importlib.import_module("loaded_app")
        with pytest.raises(ValueError, match="already imported"):
            enable("loaded_app")
    finally:
        sys.modules.pop("loaded_app", None)


def test_stale_bytecode_does_not_bypass_hook(tmp_path, monkeypatch, model):
    (tmp_path / "cached_app.py").write_text("result = 2 + 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert importlib.import_module("cached_app").result == 5
        sys.modules.pop("cached_app")
        with enable("cached_app"):
            assert importlib.import_module("cached_app").result == 5
        assert model.calls == [(OpCode.ADD, 2, 3)]
        sys.modules.pop("cached_app")
        model.calls.clear()
        importlib.import_module("cached_app")
        assert not model.calls
    finally:
        sys.modules.pop("cached_app", None)


def test_match_patterns_stay_native_but_guards_use_alu(model):
    ns = execute('''
def select(value):
    match value:
        case -1:
            return "negative"
        case n if n > 0:
            return "positive"
result = select(2)
''')
    assert ns["result"] == "positive"
    assert model.calls == [(OpCode.GT, 2, 0), (OpCode.BOOL, True, None)]


def test_runtime_alias_does_not_collide_with_application_definitions(model):
    ns = execute('''
def _allmengine_runtime():
    pass
result = 2 + 3
''')
    assert ns["result"] == 5


def test_augmented_private_attribute_respects_class_name_mangling(model):
    ns = execute('''
class Counter:
    def __init__(self):
        self.__value = 1
    def increment(self):
        self.__value += 1
        return self.__value
result = Counter().increment()
''')
    assert ns["result"] == 2
    assert model.calls == [(OpCode.ADD, 1, 1)]


def test_trainer_instruments_plain_application_but_not_test_oracle(tmp_path):
    (tmp_path / "plain_app.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_app.py").write_text(
        "from plain_app import add\n"
        "def test_add():\n    assert add(3, 4) == 3 + 4\n")
    feedback = Trainer("allmengine.reference:ReferenceModel", tmp_path,
                       instrument=("plain_app",)).evaluate(ReferenceModel())
    assert feedback.success
    # A network with a deliberately wrong ADD result must fail the native oracle.
    from allmengine import BinaryNetwork
    model = BinaryNetwork()
    state = model.state_dict()
    for row in state["layers"][-1]["weights"]:
        row[:] = [0] * len(row)
    state["layers"][-1]["biases"][:] = [-1] * model.config.output_bits
    model.load_state_dict(state)
    feedback = Trainer("allmengine.network:BinaryNetwork", tmp_path,
                       instrument=("plain_app",)).evaluate(model)
    assert feedback.failed == 1


def test_module_runner_instruments_entrypoint_and_passes_arguments(tmp_path):
    (tmp_path / "entry_app.py").write_text('''
import sys
from allmengine import alu
class Model:
    def forward(self, opcode, lhs, rhs=None):
        return 999
alu.set_model(Model())
print(2 + 3)
print(sys.argv[1:])
''')
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(source_root)
    process = subprocess.run([sys.executable, "-m", "allmengine", "entry_app", "--example"],
                             cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["999", "['--example']"]
