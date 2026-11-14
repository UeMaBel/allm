import pytest
from allmengine import NeuralValue

from addition_cases import TEST_POSITIONS as ADDITION_CASES
from subtraction_cases import TEST_POSITIONS as SUBTRACTION_CASES
from multiplication_cases import TEST_POSITIONS as MULTIPLICATION_CASES
from division_cases import TEST_POSITIONS as DIVISION_CASES
from mod_cases import TEST_POSITIONS as MOD_CASES


@pytest.mark.parametrize("case", ADDITION_CASES.values(), ids=ADDITION_CASES.keys())
def test_addition(case):
    a, b = NeuralValue(case["a"]), NeuralValue(case["b"])
    result = a + b
    assert result.value == case["is"]
    assert result == case["is"]


@pytest.mark.parametrize("case", SUBTRACTION_CASES.values(), ids=SUBTRACTION_CASES.keys())
def test_subtraction(case):
    a, b = NeuralValue(case["a"]), NeuralValue(case["b"])
    result = a - b
    assert result.value == case["is"]
    assert result == case["is"]


@pytest.mark.parametrize("case", MULTIPLICATION_CASES.values(), ids=MULTIPLICATION_CASES.keys())
def test_multiplication(case):
    a, b = NeuralValue(case["a"]), NeuralValue(case["b"])
    result = a * b
    assert result.value == case["is"]
    assert result == case["is"]


@pytest.mark.parametrize("case", DIVISION_CASES.values(), ids=DIVISION_CASES.keys())
def test_division(case):
    a, b = NeuralValue(case["a"]), NeuralValue(case["b"])
    result = a / b
    assert result.value == case["is"]
    assert result == case["is"]


@pytest.mark.parametrize("case", MOD_CASES.values(), ids=MOD_CASES.keys())
def test_mod(case):
    a, b = NeuralValue(case["a"]), NeuralValue(case["b"])
    result = a % b
    assert result.value == case["is"]
    assert result == case["is"]


@pytest.mark.parametrize("a", [0, 1, -1, 0.5])
def test_division_by_zero(a):
    with pytest.raises(ZeroDivisionError):
        NeuralValue(a) / 0


@pytest.mark.parametrize("a", [0, 1, -1, 0.5])
def test_modulo_by_zero(a):
    with pytest.raises(ZeroDivisionError):
        NeuralValue(a) % 0
