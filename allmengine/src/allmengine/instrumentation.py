"""Opt-in source instrumentation for application modules imported after enable()."""
import ast
import importlib.abc
import importlib.machinery
import keyword
import operator
import sys

from .cpu import OpCode, alu


_BINARY = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow", ast.MatMult: "matmul",
    ast.LShift: "lshift", ast.RShift: "rshift", ast.BitAnd: "and_",
    ast.BitOr: "or_", ast.BitXor: "xor",
}
_UNARY = {ast.UAdd: "pos", ast.USub: "neg", ast.Invert: "invert"}
_COMPARE = {ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne",
            ast.Gt: "gt", ast.GtE: "ge"}


def _numeric(value):
    # The ALU dispatches floats natively and integers to the network.
    # Custom classes retain their own operator behavior.
    return type(value) in (bool, int, float)


def _binary(name, left, right):
    if _numeric(left) and _numeric(right):
        return alu.execute(OpCode(name), left, right)
    return getattr(operator, name)(left, right)


def _inplace(name, left, right):
    if _numeric(left) and _numeric(right):
        return alu.execute(OpCode(name), left, right)
    return getattr(operator, "i" + name.rstrip("_"))(left, right)


def _unary(name, value):
    if _numeric(value):
        return alu.execute(OpCode(name), value)
    return getattr(operator, name)(value)


def _truth(value):
    if _numeric(value):
        result = alu.execute(OpCode.BOOL, value)
        if type(result) is not bool:
            raise TypeError("The ALU BOOL operation must return a Python bool")
        return result
    return bool(value)


class _TruthValue:
    """Let Python itself handle short-circuiting and operand evaluation order."""
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return _truth(self.value)


class _CompareOperand:
    def __init__(self, value):
        self.value = value


def _comparison(name):
    def compare(self, other):
        return _TruthValue(_binary(name, self.value, other.value))
    return compare


for _name in _COMPARE.values():
    setattr(_CompareOperand, f"__{_name}__", _comparison(_name))


class _AugmentedTarget:
    def __init__(self, owner, key, attribute):
        self.owner, self.key, self.attribute = owner, key, attribute
        self.value = getattr(owner, key) if attribute else owner[key]

    def apply(self, name, right):
        value = _inplace(name, self.value, right)
        if self.attribute:
            setattr(self.owner, self.key, value)
        else:
            self.owner[self.key] = value


class _Transform(ast.NodeTransformer):
    def __init__(self, alias):
        self.alias = alias
        self.class_name = ""

    def visit_ClassDef(self, node):
        previous = self.class_name
        self.class_name = node.name.lstrip("_")
        try:
            return self.generic_visit(node)
        finally:
            self.class_name = previous

    def call(self, method, *args):
        return ast.Call(ast.Attribute(ast.Name(self.alias, ast.Load()), method, ast.Load()),
                        list(args), [])

    def _compare(self, node):
        return ast.Compare(self.call("_CompareOperand", self.visit(node.left)), node.ops,
                           [self.call("_CompareOperand", self.visit(value))
                            for value in node.comparators])

    def condition(self, node):
        # Keep native short-circuit structure in conditional contexts. This
        # avoids testing a selected operand twice (important for stateful bool).
        if isinstance(node, ast.BoolOp):
            return ast.copy_location(ast.BoolOp(node.op, [self.condition(v) for v in node.values]), node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(ast.Not(), self.condition(node.operand)), node)
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            return ast.copy_location(self._compare(node), node)
        return ast.copy_location(self.call("_truth", self.visit(node)), node)

    def visit_BinOp(self, node):
        return ast.copy_location(self.call("_binary", ast.Constant(_BINARY[type(node.op)]),
                                           self.visit(node.left), self.visit(node.right)), node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(ast.Not(), self.condition(node.operand)), node)
        return ast.copy_location(self.call("_unary", ast.Constant(_UNARY[type(node.op)]),
                                           self.visit(node.operand)), node)

    def visit_BoolOp(self, node):
        selected = ast.BoolOp(node.op, [self.call("_TruthValue", self.visit(v)) for v in node.values])
        return ast.copy_location(ast.Attribute(selected, "value", ast.Load()), node)

    def visit_Compare(self, node):
        # Identity/membership must see original objects, not adapter objects.
        # Mixed chains containing them deliberately retain native comparisons.
        if not all(type(op) in _COMPARE for op in node.ops):
            return self.generic_visit(node)
        return ast.copy_location(ast.Attribute(self._compare(node), "value", ast.Load()), node)

    def visit_If(self, node):
        # Entry-point guards control Python module startup, not application math.
        test = node.test
        main_guard = (
            isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and ((isinstance(test.left, ast.Name) and test.left.id == "__name__"
                  and isinstance(test.comparators[0], ast.Constant)
                  and test.comparators[0].value == "__main__")
                 or (isinstance(test.left, ast.Constant) and test.left.value == "__main__"
                     and isinstance(test.comparators[0], ast.Name)
                     and test.comparators[0].id == "__name__"))
        )
        if not main_guard:
            node.test = self.condition(node.test)
        node.body = [self.visit(stmt) for stmt in node.body]
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        return node

    visit_While = visit_If

    def visit_IfExp(self, node):
        return ast.copy_location(ast.IfExp(self.condition(node.test), self.visit(node.body),
                                           self.visit(node.orelse)), node)

    def visit_Assert(self, node):
        node.test = self.condition(node.test)
        if node.msg:
            node.msg = self.visit(node.msg)
        return node

    def visit_comprehension(self, node):
        node.iter = self.visit(node.iter)
        node.ifs = [self.condition(value) for value in node.ifs]
        return node

    def visit_match_case(self, node):
        # Pattern matching has its own literal grammar and native semantics.
        # Only a case guard is an ordinary truthiness check.
        if node.guard is not None:
            node.guard = self.condition(node.guard)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_AugAssign(self, node):
        name = ast.Constant(_BINARY[type(node.op)])
        right = self.visit(node.value)
        if isinstance(node.target, ast.Name):
            value = self.call("_inplace", name, ast.Name(node.target.id, ast.Load()), right)
            return ast.copy_location(ast.Assign([node.target], value), node)
        if isinstance(node.target, ast.Attribute):
            owner = self.visit(node.target.value)
            attr = node.target.attr
            if self.class_name and attr.startswith("__") and not attr.endswith("__"):
                attr = "_" + self.class_name + attr
            key = ast.Constant(attr)
            attribute = True
        else:
            owner = self.visit(node.target.value)
            key = self._slice(node.target.slice)
            attribute = False
        target = self.call("_AugmentedTarget", owner, key, ast.Constant(attribute))
        call = ast.Call(ast.Attribute(target, "apply", ast.Load()), [name, right], [])
        return ast.copy_location(ast.Expr(call), node)

    def _slice(self, node):
        if isinstance(node, ast.Slice):
            # Avoid reliance on an application name `slice`, which may be shadowed.
            return self.call("_slice", *(self.visit(v) if v is not None else ast.Constant(None)
                                         for v in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Tuple):
            return ast.Tuple([self._slice(v) for v in node.elts], ast.Load())
        return self.visit(node)


_slice = slice


def _compile(source, filename):
    tree = ast.parse(source, filename)
    # A single leading underscore avoids class-private name mangling.
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    names.update(node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    names.update(node.asname or node.name.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.alias))
    names.update(node.name for node in ast.walk(tree)
                 if isinstance(node, ast.ExceptHandler) and node.name)
    alias = "_allmengine_runtime"
    while alias in names:
        alias += "_"
    tree = _Transform(alias).visit(tree)
    runtime_import = ast.Import([ast.alias("allmengine.instrumentation", alias)])
    position = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        position = 1
    while position < len(tree.body) and isinstance(tree.body[position], ast.ImportFrom) \
            and tree.body[position].module == "__future__":
        position += 1
    tree.body.insert(position, runtime_import)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec")


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Always compile source: ordinary and instrumented bytecode must never
        # share a .pyc cache. Original paths/line numbers remain in tracebacks.
        return _compile(self.get_data(self.path), self.path)


class Instrumentation(importlib.abc.MetaPathFinder):
    """Import hook handle. close() stops instrumentation of future imports."""
    def __init__(self, packages):
        self.packages = packages

    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == name or fullname.startswith(name + ".") for name in self.packages):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError(f"allmengine requires Python source for {fullname}")
        spec.loader = _Loader(fullname, spec.origin)
        return spec

    def close(self):
        if self in sys.meta_path:
            sys.meta_path.remove(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def enable(*packages):
    """Instrument explicitly selected application packages before importing them.

    Example: enable("myapp"); from myapp.main import main; main()
    No global Python monkey-patching and no manual NeuralValue wrappers.
    """
    if not packages:
        raise ValueError("Select at least one application package or module")
    blocked = set(sys.stdlib_module_names) | {"allmengine", "pytest", "_pytest"}
    for name in packages:
        if not isinstance(name, str) or not all(part.isidentifier() and not keyword.iskeyword(part)
                                               for part in name.split(".")):
            raise ValueError("Expected dotted application module names")
        if name.split(".")[0] in blocked:
            raise ValueError(f"Cannot instrument engine, test infrastructure, or standard library: {name}")
        if any(loaded == name or loaded.startswith(name + ".") for loaded in sys.modules):
            raise ValueError(f"{name} is already imported; enable instrumentation before importing it")
        if any(isinstance(hook, Instrumentation) and any(
                name == selected or name.startswith(selected + ".") or selected.startswith(name + ".")
                for selected in hook.packages) for hook in sys.meta_path):
            raise ValueError(f"Instrumentation already enabled for {name}")
    hook = Instrumentation(tuple(packages))
    sys.meta_path.insert(0, hook)
    return hook
